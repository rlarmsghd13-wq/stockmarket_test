"""L4 — 전략 스코어. 세 프롬프트를 계산 가능한 규칙으로.

세 프롬프트의 공통 요구는 **"단순히 X가 높다/낮다는 이유만으로 매수하지 않는다"**이다.
가중치만으로는 이걸 구현할 수 없어서 두 장치를 쓴다.

  게이트 — 못 넘으면 점수와 무관하게 탈락 ("좋은 기업"의 최소 조건)
  감점   — 주요 지표가 좋아도 질이 나쁘면 깎는다 (밸류트랩·성장 둔화 탐지)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VALUE, GROWTH = "value", "growth"

# 2018-06~2026-06, 97개월 · 425종목(생존편향 제거) 검증 결과에 맞춘 가중치.
# 부문별 IC(3개월 순방향 수익률과의 순위상관):
#     밸류에이션 +0.061  수익성 +0.033  성장성 +0.015  안정성 +0.006  모멘텀 -0.003
#
# 급등주 전략은 제거했다. 종합점수 IC가 +0.012로 예측력이 없었고,
# 주 가중치였던 모멘텀 부문이 -0.003에 기간 분할에서 부호까지 뒤집혔다.
#
# 안정성은 IC가 0에 가깝지만 10%를 남긴다. **수익률을 예측하려고 넣은 게 아니라
# 망하는 회사를 피하려고 넣은 항목**이기 때문이다. IC는 그 역할을 재지 못한다.
WEIGHTS: dict[str, dict[str, float]] = {
    VALUE:  {"valuation": 50, "profitability": 30, "stability": 10,
             "growth": 10, "momentum": 0},
    GROWTH: {"valuation": 35, "profitability": 30, "growth": 25,
             "stability": 10, "momentum": 0},
}

LABEL = {VALUE: "가치투자", GROWTH: "성장주"}


# ---------------------------------------------------------------------------
# 게이트 판정에 필요한 이력 특성
# ---------------------------------------------------------------------------
def gate_features(ttm: pd.DataFrame, asof: str) -> pd.DataFrame:
    """종목별 이력 기반 게이트 특성. 시점 잠금을 지킨다."""
    vis = ttm[pd.to_datetime(ttm["rcept_dt"]) <= pd.Timestamp(asof)]
    rows = []
    for t, g in vis.groupby("ticker"):
        g = g.sort_values("period")
        annual = g[g["quarter"] == 4].tail(3)
        op = annual.get("operating_income_ttm", pd.Series(dtype=float))
        fcf = annual.get("fcf_ttm", pd.Series(dtype=float))
        rev = g.get("revenue_ttm", pd.Series(dtype=float))

        # 최근 4개 분기 TTM YoY 성장 횟수
        gq = 0
        for k in range(1, 5):
            if len(g) > k + 3:
                cur, prev = g["revenue_ttm"].iloc[-k], g["revenue_ttm"].iloc[-k - 4]
                if pd.notna(cur) and pd.notna(prev) and prev > 0 and cur > prev:
                    gq += 1

        # 3년 연속 매출 역성장
        ann_rev = annual.get("revenue_ttm", pd.Series(dtype=float)).dropna()
        decline3 = bool(len(ann_rev) >= 3 and (ann_rev.diff().dropna() < 0).all())

        # 이익의 질 — 영업CF/순이익 50% 미만이 2년 연속
        ocf_bad = 0
        if {"ocf_ttm", "net_income_ttm"}.issubset(annual.columns):
            for _, r in annual.tail(2).iterrows():
                ni, ocf = r.get("net_income_ttm"), r.get("ocf_ttm")
                if pd.notna(ni) and pd.notna(ocf) and ni > 0 and ocf / ni < 0.5:
                    ocf_bad += 1

        rows.append({
            "ticker": t,
            "op_loss_years_3y": int((op.dropna() < 0).sum()),
            "fcf_pos_years_3y": int((fcf.dropna() > 0).sum()),
            "annual_count": int(len(annual)),
            "rev_growth_q_of_4": gq,
            "revenue_decline_3y": decline3,
            "ocf_quality_bad_2y": ocf_bad >= 2,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 게이트
# ---------------------------------------------------------------------------
def _gates_value(r) -> list[str]:
    """가치투자 — "기업의 질과 저평가를 모두 충족하는 종목을 우선한다"."""
    f = []
    if r.get("annual_count", 0) >= 3 and r.get("op_loss_years_3y", 0) > 1:
        f.append(f"3년 중 영업적자 {r['op_loss_years_3y']}회")
    dr = r.get("debt_ratio")
    if pd.notna(dr) and dr > 200:
        f.append(f"부채비율 {dr:.0f}%")
    if r.get("annual_count", 0) >= 3 and r.get("fcf_pos_years_3y", 0) < 2:
        f.append(f"3년 중 FCF 흑자 {r['fcf_pos_years_3y']}회")
    roe = r.get("roe")
    if pd.notna(roe) and roe < 8:
        f.append(f"ROE {roe:.1f}%")
    if r.get("market_cap", 0) < 5_000e8:
        f.append("시총 5천억 미만")
    # 교차 조건 — 싸기만 하거나 좋기만 한 종목은 뺀다
    v, q = r.get("valuation_score"), r.get("quality_score")
    if pd.notna(v) and v < 60:
        f.append("밸류 상위 40% 밖")
    if pd.notna(q) and q < 50:
        f.append("퀄리티 상위 50% 밖")
    return f


def _gates_growth(r) -> list[str]:
    """성장주 — "성장의 지속 가능성과 현재 밸류에이션을 함께 고려한다"."""
    f = []
    # 3년 매출 CAGR 조건을 뺐다 — 검증에서 IC -0.029로 **역방향**이었다.
    # 과거 성장이 높을수록 이후 수익률이 낮았다.
    # 대신 성장성 부문 상위 20%를 요구한다. IC는 낮지만(+0.015)
    # 5분위로 보면 최상위만 뚜렷이 좋았다 (Q5 7.40% vs Q1 5.26%).
    gs = r.get("growth_score")
    if pd.isna(gs) or gs < 80:
        f.append(f"성장성 {gs:.0f}점 (상위 20% 미달)" if pd.notna(gs) else "성장성 점수 없음")
    if r.get("rev_growth_q_of_4", 0) < 3:
        f.append(f"최근 4분기 중 성장 {r.get('rev_growth_q_of_4', 0)}회")
    cr = r.get("current_ratio")
    if pd.notna(cr) and cr < 100:
        f.append(f"유동비율 {cr:.0f}%")
    if r.get("market_cap", 0) < 3_000e8:
        f.append("시총 3천억 미만")
    return f


# ---------------------------------------------------------------------------
# 감점
# ---------------------------------------------------------------------------
def _penalty(strategy: str, r) -> tuple[float, list[str]]:
    p, why = 0.0, []
    if strategy == VALUE:
        if r.get("revenue_decline_3y"):
            p -= 15; why.append("3년 연속 매출 역성장 −15")
        if r.get("ocf_quality_bad_2y"):
            p -= 15; why.append("영업CF/순이익 50% 미만 2년 −15")
        pbr, roe = r.get("pbr"), r.get("roe")
        if pd.notna(pbr) and pd.notna(roe) and pbr < 0.5 and roe < 5:
            p -= 20; why.append("밸류트랩 (PBR<0.5 & ROE<5%) −20")
    elif strategy == GROWTH:
        acc = r.get("growth_acceleration")
        if pd.notna(acc) and acc < -10:
            p -= 15; why.append(f"성장 둔화 {acc:.0f}%p −15")
        peg = r.get("peg")
        if pd.notna(peg) and peg >= 3:
            p -= 20; why.append(f"PEG {peg:.1f} −20")
        elif pd.notna(peg) and peg <= 1.0:
            p += 5; why.append(f"PEG {peg:.1f} +5")
    # 공통 — 구조변경 구간의 TTM은 믿을 수 없다
    if r.get("ttm_reliable") is False:
        p -= 10; why.append("구조변경 직후 (TTM 신뢰 불가) −10")
    return p, why


def _removed_momentum_thresholds(df: pd.DataFrame) -> dict:
    """급등 게이트 임계값을 유니버스 분포에서 잡는다.

    설계 초안의 고정값(1개월 +20%, 거래량비 2.0)은 중소형주 기준이라
    대형주 200종목에서는 통과 종목이 거의 없다 — 실측 거래량비 중앙값이 1.0 안팎이다.
    분포 상위 백분위로 바꿔 유니버스에 맞춘다.
    """
    r1 = df["return_1m"].dropna()
    vr = df["volume_ratio"].dropna()
    return {
        "return_1m": float(r1.quantile(0.80)) if len(r1) else 20.0,
        "volume_ratio": float(vr.quantile(0.80)) if len(vr) else 2.0,
    }


def score(df: pd.DataFrame, strategy: str, thresholds: dict | None = None) -> pd.DataFrame:
    """부문 점수 + 게이트 + 감점 → 전략 점수."""
    w = WEIGHTS[strategy]
    out = []
    for _, r in df.iterrows():
        d = r.to_dict()
        base, wsum = 0.0, 0.0
        for cat, weight in w.items():
            s = d.get(f"{cat}_score")
            if weight and pd.notna(s):
                base += s * weight
                wsum += weight
        base = base / wsum if wsum else np.nan

        if strategy == VALUE:
            fails = _gates_value(d)
        else:
            fails = _gates_growth(d)

        pen, why = _penalty(strategy, d)
        out.append({
            "ticker": d["ticker"], "name": d["name"], "track": d["track"],
            "strategy": strategy,
            "base_score": base,
            "penalty": pen,
            "score": (base + pen) if pd.notna(base) else np.nan,
            "gate_pass": len(fails) == 0,
            "gate_fails": " · ".join(fails),
            "penalty_why": " · ".join(why),
            "coverage": int(sum(pd.notna(d.get(f"{c}_score")) for c in w if w[c])),
        })
    res = pd.DataFrame(out)
    return res.sort_values("score", ascending=False).reset_index(drop=True)
