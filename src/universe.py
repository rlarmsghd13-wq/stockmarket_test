"""L0 — 유니버스 구성.

핵심 원칙: **제외 종목도 행을 남긴다.**
행을 지우면 그 시점에 존재했던 종목이 사라져 백테스트에 생존편향이 생긴다.
`included` 불리언과 `excluded_reason`으로만 표시한다.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

import config
from . import krx


def _liquidity(d: str, market: str, sample_days: int = 5) -> pd.DataFrame:
    """리밸런싱일 직전 영업일 몇 개의 평균 거래대금.

    60일 전부를 받으면 호출이 너무 많아진다. 대형주 유동성 필터는
    거의 걸리지 않는 조건이라 5영업일 표본이면 충분하다.
    """
    cur = date.fromisoformat(d)
    frames, got = [], 0
    for _ in range(20):
        if got >= sample_days:
            break
        snap = krx.market_snapshot(krx.ymd(cur), market)
        if krx.is_settled(snap):
            frames.append(snap[["ticker", "trading_value"]])
            got += 1
        cur -= timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["ticker", "avg_trading_value"])
    out = (pd.concat(frames).groupby("ticker", as_index=False)["trading_value"].mean()
           .rename(columns={"trading_value": "avg_trading_value"}))
    return out


def snapshot(asof: str) -> pd.DataFrame:
    """한 시점의 전 종목 + 유니버스 편입 여부."""
    bd = krx.prev_business_day(asof)   # 미확정 당일·휴장일이면 직전 영업일
    frames = []
    for market in config.MARKETS:
        cap = krx.market_snapshot(bd, market)
        if cap.empty:
            continue
        names = krx.ticker_names(bd, market)
        liq = _liquidity(asof, market)
        df = cap.merge(names, on="ticker", how="left").merge(liq, on="ticker", how="left")
        frames.append(df)
    if not frames:
        raise RuntimeError(f"{asof} 시점 시장 데이터를 가져오지 못했습니다.")

    all_df = pd.concat(frames, ignore_index=True)
    all_df["asof_date"] = asof
    all_df["price_date"] = bd     # 실제로 시세를 가져온 영업일

    # 우선주 제외 — 보통주와 재무제표를 공유해 지표가 중복된다.
    all_df["is_preferred"] = ~all_df["ticker"].str.endswith("0")

    reasons: list[str | None] = []
    for r in all_df.itertuples():
        if r.is_preferred:
            reasons.append("우선주")
        elif pd.isna(r.market_cap) or r.market_cap <= 0:
            reasons.append("시총 없음")
        elif r.market_cap < config.MIN_MARKET_CAP:
            reasons.append("시총 미달")
        elif pd.isna(r.avg_trading_value) or r.avg_trading_value < config.MIN_AVG_TRADING_VALUE:
            reasons.append("거래대금 미달")
        else:
            reasons.append(None)
    all_df["excluded_reason"] = reasons

    # 남은 종목 중 시총 상위 N만 편입. 나머지는 사유를 남긴다.
    eligible = all_df[all_df["excluded_reason"].isna()].copy()
    eligible = eligible.sort_values("market_cap", ascending=False)
    keep = set(eligible.head(config.UNIVERSE_SIZE)["ticker"])

    all_df["included"] = all_df["ticker"].isin(keep)
    all_df.loc[
        all_df["excluded_reason"].isna() & ~all_df["included"],
        "excluded_reason",
    ] = f"시총 상위 {config.UNIVERSE_SIZE} 밖"

    all_df["cap_rank"] = all_df["market_cap"].rank(ascending=False, method="min")
    cols = ["asof_date", "price_date", "ticker", "name", "market", "close", "market_cap",
            "shares_out", "volume", "trading_value", "avg_trading_value",
            "cap_rank", "included", "excluded_reason"]
    return all_df[[c for c in cols if c in all_df.columns]]


def build_history(dates: list[str] | None = None) -> pd.DataFrame:
    """리밸런싱 시점별 유니버스 스냅샷을 전부 쌓는다.

    상장폐지 종목은 폐지 이전 시점의 스냅샷에 자연히 포함되므로
    별도 처리 없이 생존편향이 제거된다.
    """
    dates = dates or krx.rebalance_dates()
    out = []
    for d in dates:
        try:
            out.append(snapshot(d))
        except Exception as exc:
            print(f"  [skip] {d}: {exc}")
    if not out:
        raise RuntimeError("유니버스 스냅샷을 하나도 만들지 못했습니다.")
    return pd.concat(out, ignore_index=True)
