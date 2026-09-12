"""L1 — DART 원문을 분기 재무 테이블로.

핵심 결정: **손익·현금흐름은 누적(cumulative) 기준으로 저장한다.**

DART는 회사·보고서마다 thstrm_amount에 3개월치를 넣기도 하고 누적을 넣기도 한다.
현금흐름표는 대부분 누적만 낸다. 단일 분기값을 원본에서 직접 읽으려 하면
회사마다 다른 규칙에 끌려다니게 된다.

누적으로 통일해두면
  - TTM = 당기 누적 + 전년 연간 − 전년 동기 누적   (차분 없이 바로 계산)
  - 단일 분기 = 당기 누적 − 직전 분기 누적
둘 다 안정적으로 나온다. 특히 4분기 손익은 원본에 없고 이 방식으로만 얻어진다.
"""
from __future__ import annotations

import re

import pandas as pd

import config
from . import accounts, dart

_NUM = re.compile(r"[^\d.\-]")


def parse_amount(v) -> float | None:
    """'1,234,567' → 1234567.0 / '-' → None / '(1,234)' → -1234.0"""
    if v is None:
        return None
    s = str(v).strip()
    if s in ("", "-", "--", "nan", "None"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = _NUM.sub("", s)
    if s in ("", "-", "."):
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    return -x if neg else x


def _cumulative(row: pd.Series, reprt_code: str) -> float | None:
    """이 행의 누적 금액.

    사업보고서의 thstrm_amount는 이미 연간 누적이다.
    분기보고서는 thstrm_add_amount(누적)를 우선하고, 없으면 thstrm_amount로 폴백한다.
    1분기는 3개월치와 누적이 같으므로 폴백이 안전하다.
    """
    if reprt_code == config.REPRT_ANNUAL:
        return parse_amount(row.get("thstrm_amount"))
    add = parse_amount(row.get("thstrm_add_amount"))
    if add is not None:
        return add
    return parse_amount(row.get("thstrm_amount"))


def report_currency(raw: pd.DataFrame) -> str:
    """보고서의 표시 통화. DART `currency` 필드의 최빈값.

    두산밥캣은 USD, 일부 외국기업(950xxx)은 USD·JPY로 공시한다. 이걸 원화로
    읽으면 재무제표 값은 그대로인데 시가총액(원)과 나누는 순간 무의미해진다.
    실제로 두산밥캣 PBR이 1,180으로 잡혀 있었다 — 1000배 단위 오류처럼 보였지만
    원인은 통화였다.
    """
    if raw.empty or "currency" not in raw.columns:
        return "KRW"
    v = raw["currency"].dropna().astype(str).str.strip()
    v = v[v != ""]
    return v.mode().iloc[0] if not v.empty else "KRW"


def parse_report(raw: pd.DataFrame, reprt_code: str) -> dict[str, float]:
    """한 보고서의 원문 행들 → {필드: 금액}.

    같은 필드에 여러 행이 매칭되면 첫 번째(가장 상위 계정)를 쓴다.
    단, CAPEX는 유형·무형이 각각 여러 줄로 나올 수 있어 합산한다.
    """
    out: dict[str, float] = {}
    if raw.empty:
        return out
    for r in raw.to_dict("records"):
        field = accounts.match_field(
            r.get("account_id", ""), r.get("account_nm", ""), r.get("sj_div", ""))
        if field is None:
            continue
        if field in accounts.STOCK_FIELDS:
            val = parse_amount(r.get("thstrm_amount"))
        else:
            val = _cumulative(pd.Series(r), reprt_code)
        if val is None:
            continue
        if field in accounts.CAPEX_FIELDS:
            out[field] = out.get(field, 0.0) + abs(val)
        elif field not in out:
            out[field] = val
    return out


def fetch_company(corp_code: str, ticker: str, years: list[int]) -> pd.DataFrame:
    """한 종목의 연도×보고서별 재무를 전부 긁어 한 행씩으로."""
    rows = []
    for year in years:
        for reprt in config.REPRT_ORDER:
            raw = dart.financials(corp_code, year, reprt, "CFS")
            fs_div = "CFS"
            if raw.empty:
                raw = dart.financials(corp_code, year, reprt, "OFS")
                fs_div = "OFS"
            if raw.empty:
                continue
            vals = parse_report(raw, reprt)
            if not vals:
                continue
            rcept = dart.filing_date(corp_code, year, reprt)
            if rcept is None:
                # 접수일을 모르면 시점 잠금이 불가능하다. 버리는 게 맞다.
                continue
            rows.append({
                "ticker": ticker, "corp_code": corp_code,
                "year": year, "quarter": config.REPRT_QUARTER_NO[reprt],
                "reprt_code": reprt, "fs_div": fs_div, "rcept_dt": rcept,
                "currency": report_currency(raw),
                **vals,
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["period"] = df["year"] * 4 + df["quarter"]   # 정렬·차분용 연속 인덱스
    return df.sort_values("period").reset_index(drop=True)


def single_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """누적값을 차분해 단일 분기 손익을 만든다. 성장 변동성 계산에 쓴다.

    1분기는 누적이 곧 분기값이고, 그 외는 직전 분기 누적을 뺀다.
    연도가 바뀌면 누적이 리셋되므로 같은 해 안에서만 차분한다.
    """
    if df.empty:
        return df
    flows = [c for c in accounts.FLOW_FIELDS if c in df.columns]
    out = df.copy().sort_values(["year", "quarter"])
    for f in flows:
        prev = out.groupby("year")[f].shift(1)
        q = out[f] - prev
        out[f"{f}_q"] = q.where(out["quarter"] != 1, out[f])
    return out.reset_index(drop=True)
