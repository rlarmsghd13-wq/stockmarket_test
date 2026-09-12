"""백테스트용 패널 준비 — 실전 유니버스 재현, 벤치마크, 월간 수익률.

scripts/16, 17이 같이 쓴다. 검증마다 유니버스 조건을 다시 적으면
어느 스크립트가 어떤 표본을 봤는지 알 수 없게 된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

from . import store


def _turnover() -> pd.DataFrame:
    """60일 평균 거래대금(근사). close_adj × volume 이므로 액면분할 구간에서
    왜곡되지만, 종목 간 유동성 순위를 보는 용도라 감수한다."""
    px = pd.concat([store.load("prices_daily_hist"), store.load("prices_daily")],
                   ignore_index=True).drop_duplicates(["ticker", "date"], keep="last")
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])
    px["tv"] = px["close_adj"] * px["volume"]
    px["avg_tv"] = (px.groupby("ticker")["tv"]
                      .transform(lambda s: s.rolling(60, min_periods=20).mean()))
    return px[["ticker", "date", "avg_tv"]].dropna()


def check_coverage(warn_drop: float = 0.5) -> list[str]:
    """시세 커버리지 점검 — 표본이 특정 시점부터 급감하지 않는지 본다.

    2026-09-10에 `prices_daily_hist`가 2023-06-30에서 끊겨 있는 것을 뒤늦게
    발견했다. 그 구간부터 표본이 423 → 190종목으로 줄어 생존편향이 되살아났고,
    벤치마크가 두 배로 부풀려져 있었다. **백테스트는 이런 식으로 조용히
    틀린 답을 낸다.** 그래서 매 실행마다 확인한다.
    """
    px = pd.concat([store.load("prices_daily_hist"), store.load("prices_daily")],
                   ignore_index=True).drop_duplicates(["ticker", "date"], keep="last")
    px["date"] = pd.to_datetime(px["date"])
    n = px.groupby(px["date"].dt.to_period("M"))["ticker"].nunique()
    peak = n.max()
    bad = n[(n < peak * warn_drop) & (n.index > n.index.min() + 6)]
    msgs = []
    if len(bad):
        msgs.append(f"[경고] 시세 종목 수가 최대 {peak}개 대비 "
                    f"{warn_drop*100:.0f}% 미만인 달이 {len(bad)}개 있습니다: "
                    f"{bad.index.min()} ~ {bad.index.max()}")
        msgs.append("       scripts/20_extend_prices.py --dry-run 으로 확인하세요.")
    return msgs


def prepare() -> pd.DataFrame:
    """gate_panel + 유동성 + 실전 유니버스 플래그 + 벤치마크 3종."""
    for m in check_coverage():
        print(m)
    gp = store.load("gate_panel")
    gp["asof_ts"] = pd.to_datetime(gp["asof"])
    gp = pd.merge_asof(gp.sort_values("asof_ts"), _turnover().sort_values("date"),
                       left_on="asof_ts", right_on="date", by="ticker",
                       direction="backward")

    gp["cap_rank"] = gp.groupby(["asof", "strategy"])["market_cap"].rank(
        ascending=False, method="min")
    gp["in_universe"] = (
        (gp["market_cap"] >= config.MIN_MARKET_CAP)
        & (gp["avg_tv"].fillna(0) >= config.MIN_AVG_TRADING_VALUE)
        & (gp["cap_rank"] <= config.UNIVERSE_SIZE))

    u = gp[gp["in_universe"]]
    eq = u.groupby(["asof", "strategy"])["fwd_63"].mean().rename("b_univ")
    cw = (u.groupby(["asof", "strategy"])
           .apply(lambda g: np.average(g["fwd_63"], weights=g["market_cap"]),
                  include_groups=False).rename("b_cap"))
    gp = gp.merge(eq, on=["asof", "strategy"], how="left")
    gp = gp.merge(cw, on=["asof", "strategy"], how="left")
    gp["exc_univ"] = gp["fwd_63"] - gp["b_univ"]
    gp["exc_cap"] = gp["fwd_63"] - gp["b_cap"]
    return gp


DELIST_GAP_DAYS = 180     # 시세가 이만큼 끊기면 상장폐지·합병으로 본다


def delisted(px: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """시세가 끝나고 다시 시작되지 않는 종목 → 마지막 거래일."""
    last = px.groupby("ticker")["date"].max()
    end = px["date"].max()
    return {t: d for t, d in last.items()
            if (end - d).days > DELIST_GAP_DAYS}


def monthly_returns(asofs: list[str], delist_haircut: float = 0.0) -> pd.DataFrame:
    """기준일 → 다음 기준일 사이의 종목별 수익률 (수정주가 기준).

    3개월 순방향(fwd_63)은 구간이 겹쳐 계좌 시뮬레이션에 쓸 수 없다.
    월 1회 리밸런싱의 실제 보유기간 수익률이 필요하다.

    delist_haircut: 상장폐지 종목의 **마지막 보유 달**에 추가로 적용할 손실률(%).
        기본 0은 "마지막 거래일 종가에 팔았다"는 가정이다. 실제로 확인된 22건은
        대부분 합병·공개매수(우리은행→우리금융, SK머티리얼즈→SK, 넥슨지티 +125%)라
        손실이 아니고, 진짜 붕괴한 4건(셀리버리 16원, 퓨처코어 15원, KH필룩스 316원,
        에이팸 135원)은 폐지 전에 이미 −94~−99%가 주가에 반영돼 있다.
        그래도 정리매매 손실이 얼마나 영향을 주는지 보려면 −30·−70·−100을 넣어본다.
    """
    px = pd.concat([store.load("prices_daily_hist"), store.load("prices_daily")],
                   ignore_index=True).drop_duplicates(["ticker", "date"], keep="last")
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values("date")

    dates = sorted(pd.Timestamp(a) for a in asofs)
    rows = []
    for t, g in px.groupby("ticker"):
        s = g.set_index("date")["close_adj"].dropna()
        if s.empty:
            continue
        # 각 기준일 이하의 마지막 종가
        idx = s.index.searchsorted(dates, side="right") - 1
        for i, (d, k) in enumerate(zip(dates, idx)):
            if k < 0 or i + 1 >= len(dates):
                continue
            k2 = idx[i + 1]
            if k2 <= k:
                continue
            # 가격이 60거래일 넘게 끊기면 상장폐지·거래정지로 본다
            if (s.index[k2] - s.index[k]).days > 120:
                continue
            rows.append({"asof": d.strftime("%Y-%m-%d"), "ticker": t,
                         "ret_m": float(s.iloc[k2] / s.iloc[k] - 1) * 100})
    out = pd.DataFrame(rows)

    if delist_haircut and not out.empty:
        # 각 상장폐지 종목의 **마지막** 수익률 행에만 감액을 곱한다.
        dead = delisted(px)
        m = out["ticker"].isin(dead)
        if m.any():
            idx = out[m].groupby("ticker")["asof"].idxmax()
            f = 1 + delist_haircut / 100
            out.loc[idx, "ret_m"] = (1 + out.loc[idx, "ret_m"] / 100) * f * 100 - 100
    return out
