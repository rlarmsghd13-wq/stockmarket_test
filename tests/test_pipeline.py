"""L1→L2 계산 체인 검증 — DART 키 없이 돌아간다.

합성 재무제표를 만들어 TTM 집계와 지표 계산이 손으로 계산한 값과 맞는지 본다.
실제 데이터로 검증하기 전에 공식 자체의 오류를 걸러낸다.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Windows 콘솔 기본 인코딩이 cp949라 한글·기호 출력이 깨진다.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import config                                    # noqa: E402
from src import accounts, metrics, quarterly, ttm  # noqa: E402

FAILS: list[str] = []


def is_null(x) -> bool:
    """결측 판정. pandas는 float 컬럼의 None을 NaN으로 바꾸므로 둘 다 결측으로 본다."""
    return x is None or (isinstance(x, float) and pd.isna(x)) or (
        hasattr(x, "dtype") and pd.isna(x))


def check(name: str, got, want, tol=1e-6):
    if is_null(want):
        ok = is_null(got)
    else:
        ok = not is_null(got) and abs(got - want) <= tol * max(1, abs(want))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got={got!r} want={want!r}")
    if not ok:
        FAILS.append(name)


def make_company(years, q_revenue=100.0, op_margin=0.20, tax=0.22,
                 equity=1000.0, assets=2000.0, liabilities=1000.0,
                 ocf_mult=1.2, capex_per_q=10.0):
    """분기마다 동일한 실적을 내는 회사. 누적 기준으로 채운다."""
    rows = []
    for y in years:
        for q, reprt in enumerate(config.REPRT_ORDER, start=1):
            cum_rev = q_revenue * q
            cum_op = cum_rev * op_margin
            cum_pre = cum_op
            cum_tax = cum_pre * tax
            cum_ni = cum_pre - cum_tax
            rows.append({
                "ticker": "TEST", "corp_code": "0000", "year": y, "quarter": q,
                "reprt_code": reprt, "fs_div": "CFS",
                "rcept_dt": f"{y}{['0515','0814','1114','0331'][q-1]}",
                "period": y * 4 + q,
                "revenue": cum_rev, "operating_income": cum_op,
                "pretax_income": cum_pre, "tax_expense": cum_tax,
                "net_income": cum_ni, "net_income_parent": cum_ni,
                "gross_profit": cum_rev * 0.4,
                "ocf": cum_ni * ocf_mult,
                "capex_tangible": capex_per_q * q, "capex_intangible": 0.0,
                "depreciation": cum_rev * 0.05, "amortization": 0.0,
                "assets": assets, "liabilities": liabilities, "equity": equity,
                "equity_parent": equity, "current_assets": 800.0,
                "current_liabilities": 400.0, "cash": 200.0,
            })
    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


print("\n[1] parse_amount — DART 문자열 파싱")
check("콤마", quarterly.parse_amount("1,234,567"), 1234567.0)
check("결측 '-'", quarterly.parse_amount("-"), None)
check("괄호 음수", quarterly.parse_amount("(1,234)"), -1234.0)
check("음수", quarterly.parse_amount("-500"), -500.0)
check("빈값", quarterly.parse_amount(""), None)

print("\n[2] 계정 매칭 — account_id 우선, 계정명 폴백")
check("표준태그 매출", 1.0 if accounts.match_field(
    "ifrs-full_Revenue", "수익(매출액)", "IS") == "revenue" else 0.0, 1.0)
check("태그없음 영업이익", 1.0 if accounts.match_field(
    "-", "영업이익", "IS") == "operating_income" else 0.0, 1.0)
check("CAPEX 계정명", 1.0 if accounts.match_field(
    "-", "유형자산의 취득", "CF") == "capex_tangible" else 0.0, 1.0)
check("무관계정 None", 0.0 if accounts.match_field(
    "-", "배당금지급", "CF") is None else 1.0, 0.0)

print("\n[3] TTM 집계 — 누적에서 역산")
df = make_company([2023, 2024, 2025, 2026])
t = ttm.build(df)
q2_2026 = t[(t.year == 2026) & (t.quarter == 2)].iloc[0]
# 누적Q2(200) + 연간2025(400) − 누적Q2 2025(200) = 400
check("매출 TTM(2026 2Q)", q2_2026["revenue_ttm"], 400.0)
check("영업이익 TTM", q2_2026["operating_income_ttm"], 80.0)
q4_2025 = t[(t.year == 2025) & (t.quarter == 4)].iloc[0]
check("4Q는 연간 그대로", q4_2025["revenue_ttm"], 400.0)
first = t[(t.year == 2023) & (t.quarter == 1)].iloc[0]
check("직전연도 없으면 None", first["revenue_ttm"], None)
check("FCF = 영업CF − CAPEX", q2_2026["fcf_ttm"], 62.4 * 1.2 - 40.0, tol=1e-3)

print("\n[4] 단일 분기 차분 — 성장 변동성용")
sq = quarterly.single_quarter(df)
row = sq[(sq.year == 2026) & (sq.quarter == 3)].iloc[0]
check("3Q 단일분기 매출", row["revenue_q"], 100.0)
row1 = sq[(sq.year == 2026) & (sq.quarter == 1)].iloc[0]
check("1Q는 누적=분기", row1["revenue_q"], 100.0)

print("\n[5] 지표 — 손으로 계산한 값과 대조")
hist = t[t.period <= q2_2026["period"]]
m = metrics.compute(q2_2026.to_dict(), market_cap=1000.0, shares_out=100.0,
                    hist=hist, track=config.TRACK_GENERAL)
check("영업이익률", m["operating_margin"], 20.0)
check("ROE", m["roe"], 62.4 / 1000 * 100, tol=1e-3)          # 순이익TTM / 자본
check("ROA", m["roa"], 62.4 / 2000 * 100, tol=1e-3)
check("부채비율", m["debt_ratio"], 100.0)
check("유동비율", m["current_ratio"], 200.0)
check("PER", m["per"], 1000 / 62.4, tol=1e-3)
check("PBR", m["pbr"], 1.0)
check("EPS", m["eps"], 62.4 / 100, tol=1e-3)
check("영업CF/순이익", m["ocf_to_ni"], 120.0, tol=1e-3)
check("성장률 0%", m["revenue_growth_yoy"], 0.0, tol=1e-9)

print("\n[6] 적자 처리 — PER은 반드시 None")
loss = make_company([2023, 2024, 2025, 2026], op_margin=-0.10)
tl = ttm.build(loss)
lrow = tl[(tl.year == 2026) & (tl.quarter == 2)].iloc[0]
ml = metrics.compute(lrow.to_dict(), market_cap=1000.0, shares_out=100.0,
                     hist=tl[tl.period <= lrow["period"]])
check("적자 PER", ml["per"], None)
check("적자 PBR은 계산됨", ml["pbr"], 1.0)
check("매출성장률은 흑자무관 계산됨", ml["revenue_growth_yoy"], 0.0, tol=1e-9)
# 전년이 적자면 성장률 부호가 뒤집혀 의미를 잃는다 → None이어야 한다.
check("전년 적자면 영업이익성장률 None", ml["op_growth_yoy"], None)
check("적자 PEG", ml["peg"], None)
check("적자 정상화PER도 None", ml["normalized_per"], None)

print("\n[7] 트랙별 무효화 — T2 금융은 부채비율·FCF 없음")
mf = metrics.compute(q2_2026.to_dict(), market_cap=1000.0, shares_out=100.0,
                     hist=hist, track=config.TRACK_FINANCE)
check("T2 부채비율", mf["debt_ratio"], None)
check("T2 FCF/순이익", mf["fcf_to_ni"], None)
check("T2 ROA는 유지", mf["roa"], 62.4 / 2000 * 100, tol=1e-3)

print("\n[8] 안전 나눗셈")
check("분모 0", metrics.div(10, 0), None)
check("분모 None", metrics.div(10, None), None)
check("분모 음수", metrics.div(10, -5), None)
check("음수 허용 옵션", metrics.div(10, -5, den_must_be_positive=False), -2.0)

print("\n" + "=" * 56)
print(f"실패 {len(FAILS)}건" + (": " + ", ".join(FAILS) if FAILS else " — 전부 통과"))
sys.exit(1 if FAILS else 0)
