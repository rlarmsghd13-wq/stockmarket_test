"""주식수 — EPS·BPS 분모.

두 가지 보정이 모두 필요하다.

1. **우선주를 더한다.** DART 순이익은 보통주·우선주 구분 없이 전체에 귀속되므로
   분모도 보통주 + 우선주여야 한다. 보통주만 쓰면 우선주가 있는 기업
   (삼성전자·현대차·LG전자 등)의 EPS가 체계적으로 과대평가된다.

2. **자기주식을 뺀다.** 자기주식은 의결권도 배당권도 없는 비유통주식이다.
   KRX 공표 EPS·BPS도 유통주식 기준이며, 자사주 비중이 큰 기업에서 차이가 크다.
   실측: SK 자사주 24.6% → 분모를 안 빼면 BPS가 32.6% 낮게 나온다
   (자사주 제외 배수 1.326 = KRX 역산값 1.326, 소수점까지 일치).
   삼성생명 10.2% → 1.114 (KRX 역산 1.114).

KRX 종목코드 규칙: 보통주는 6번째 자리가 0, 우선주는 5/7/9 등.
앞 5자리가 같으면 같은 발행사로 본다.
"""
from __future__ import annotations

import pandas as pd


def total_shares(snapshot: pd.DataFrame) -> pd.DataFrame:
    """유니버스 스냅샷 → 보통주 티커별 (보통주 주식수, 우선주 포함 총주식수, 총시총).

    snapshot: universe.snapshot() 결과 (우선주 행도 포함되어 있어야 한다)
    """
    df = snapshot[["ticker", "shares_out", "market_cap"]].copy()
    df["root"] = df["ticker"].str[:5]
    df["is_common"] = df["ticker"].str.endswith("0")

    agg = df.groupby("root", as_index=False).agg(
        shares_total=("shares_out", "sum"),
        market_cap_total=("market_cap", "sum"),
    )
    common = df[df["is_common"]].rename(
        columns={"shares_out": "shares_common", "market_cap": "market_cap_common"})
    out = common.merge(agg, on="root", how="left")
    out["has_preferred"] = out["shares_total"] > out["shares_common"]
    return out[["ticker", "shares_common", "shares_total",
                "market_cap_common", "market_cap_total", "has_preferred"]]


def with_treasury(sh: pd.DataFrame, corp_map: dict[str, str], year: int,
                  *, verbose: bool = False) -> pd.DataFrame:
    """`total_shares` 결과에 자기주식을 붙여 유통주식수를 만든다.

    corp_map: {ticker: corp_code}
    자기주식 조회에 실패하면 `treasury=None`으로 두고 유통주식수는 상장주식수와 같게 둔다.
    추정해서 채우지 않는다 — 못 뺀 종목은 `treasury_known=False`로 드러난다.
    """
    from . import dart   # 순환 참조 방지를 위해 지연 임포트

    treas, known = [], []
    for tk in sh["ticker"]:
        code = corp_map.get(tk)
        t = None
        if code:
            try:
                t = dart.treasury_shares(code, year)
            except Exception as exc:
                if verbose:
                    print(f"  [자기주식 조회 실패] {tk}: {exc}")
        treas.append(t or 0)
        known.append(t is not None)

    out = sh.copy()
    out["treasury"] = treas
    out["treasury_known"] = known
    out["shares_outstanding"] = (out["shares_total"] - out["treasury"]).clip(lower=1)
    out["treasury_ratio"] = out["treasury"] / out["shares_total"]
    return out
