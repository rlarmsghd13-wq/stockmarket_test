"""L1 — TTM(최근 4개 분기) 집계.

  손익·현금흐름 : TTM(Y,Q) = 누적(Y,Q) + 연간(Y−1) − 누적(Y−1,Q)
                  TTM(Y,4) = 연간(Y)
  재무상태표     : 해당 분기말 값 그대로 (시점값이라 합산하지 않는다)
  혼합 지표 분모 : (당분기말 + 4분기전) / 2 평균

연간 데이터만 쓰면 최악의 경우 15개월 묵은 숫자로 판단하게 된다. TTM이 그걸 막는다.
"""
from __future__ import annotations

import pandas as pd

from . import accounts


def build(df: pd.DataFrame) -> pd.DataFrame:
    """`quarterly.fetch_company` 결과 → TTM 테이블."""
    if df.empty:
        return df

    flows = [c for c in accounts.FLOW_FIELDS if c in df.columns]
    stocks = [c for c in accounts.STOCK_FIELDS if c in df.columns]

    d = df.sort_values(["year", "quarter"]).reset_index(drop=True)
    annual = (d[d["quarter"] == 4].set_index("year")[flows]
              if flows else pd.DataFrame())
    cum_by_yq = d.set_index(["year", "quarter"])[flows] if flows else pd.DataFrame()

    rows = []
    for r in d.itertuples():
        rec = {
            "ticker": r.ticker, "corp_code": r.corp_code,
            "year": r.year, "quarter": r.quarter, "period": r.period,
            "reprt_code": r.reprt_code, "fs_div": r.fs_div,
            "rcept_dt": r.rcept_dt,
            # 표시 통화. 원화가 아니면 시가총액과 섞는 지표를 쓸 수 없다
            "currency": getattr(r, "currency", "KRW") or "KRW",
        }

        for f in flows:
            cur = getattr(r, f, None)
            if pd.isna(cur) if cur is not None else True:
                rec[f"{f}_ttm"] = None
                continue
            if r.quarter == 4:
                rec[f"{f}_ttm"] = cur          # 연간 누적이 곧 TTM
                continue
            try:
                prev_annual = annual.at[r.year - 1, f]
                prev_cum = cum_by_yq.at[(r.year - 1, r.quarter), f]
            except KeyError:
                rec[f"{f}_ttm"] = None         # 직전 연도가 없으면 계산 불가
                continue
            if pd.isna(prev_annual) or pd.isna(prev_cum):
                rec[f"{f}_ttm"] = None
            else:
                rec[f"{f}_ttm"] = cur + prev_annual - prev_cum

        for s in stocks:
            rec[s] = getattr(r, s, None)

        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("period").reset_index(drop=True)

    # 혼합 지표용 평균 잔액 — 4분기 전 값과의 평균.
    for s in ("equity", "equity_parent", "assets"):
        if s in out.columns:
            out[f"{s}_avg"] = (out[s] + out[s].shift(4)) / 2
            out[f"{s}_avg"] = out[f"{s}_avg"].fillna(out[s])

    # 파생 합계.
    # 금융업(은행·증권·지주)은 CAPEX·감가상각 계정 자체가 없어 컬럼이 통째로 빠진다.
    # `out.get(col, 0)`은 그럴 때 Series가 아니라 int 0을 돌려주므로 .fillna()가 터진다.
    def col(name: str) -> pd.Series:
        if name in out.columns:
            return out[name].fillna(0)
        return pd.Series(0.0, index=out.index)

    has_capex = any(c in out.columns for c in ("capex_tangible_ttm", "capex_intangible_ttm"))
    if has_capex:
        out["capex_ttm"] = col("capex_tangible_ttm") + col("capex_intangible_ttm")
        out.loc[out["capex_ttm"] == 0, "capex_ttm"] = None
        if "ocf_ttm" in out.columns:
            out["fcf_ttm"] = out["ocf_ttm"] - out["capex_ttm"]

    if any(c in out.columns for c in ("depreciation_ttm", "amortization_ttm")):
        da = col("depreciation_ttm") + col("amortization_ttm")
        out["da_ttm"] = da.replace(0, None)

    return out
