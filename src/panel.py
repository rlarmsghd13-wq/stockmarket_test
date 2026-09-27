"""과거 시점별 지표·등급 패널 생성.

특정 기준일에 **그 시점에 알 수 있었던 값만으로** 전 종목 지표와 부문 점수를 만든다.
스타일 스프레드(src/style.py)와 지표 검증(scripts/09_validate.py)이 같이 쓴다.

시가총액은 그 시점 수정주가 × 현재 유통주식수로 근사한다. 과거 주식수 이력이
없어 자사주 소각·증자는 반영되지 않는다 — 밸류에이션 절대값보다 **종목 간 순위**를
보는 용도이므로 감수한다.
"""
from __future__ import annotations

import pandas as pd

from . import grading, metrics, momentum


def shares_asof(snapshots: pd.DataFrame, asof: str) -> dict[str, float]:
    """기준일 시점의 종목별 상장주식수.

    유니버스 스냅샷(과거 + 현재)에서 그 시점 이전 마지막 관측치를 쓴다.
    현재 유니버스의 주식수만 쓰면 **지금 상위 200에 없는 종목이 통째로 빠져**
    생존편향이 그대로 남는다. 실제로 그 실수로 검증 표본이 444 → 199종목으로
    줄어든 적이 있다.
    """
    d = snapshots[["ticker", "asof_date", "shares_out"]].dropna()
    d = d[d["asof_date"] <= asof]
    if d.empty:
        return {}
    d = d.sort_values("asof_date").groupby("ticker")["shares_out"].last()
    return {t: float(v) for t, v in d.items() if v and v > 0}


def cohort(asof: str, ttm: pd.DataFrame, px: pd.DataFrame,
           shares: dict[str, float] | pd.DataFrame, tracks: pd.DataFrame,
           sector: dict[str, str]) -> pd.DataFrame | None:
    """한 기준일의 전 종목 지표 + 부문 점수.

    shares: {ticker: 주식수} 또는 shares_outstanding 컬럼을 가진 DataFrame.
    tracks에 없는 종목은 기본 트랙(T1)으로 본다 — 과거 유니버스 종목은
    현재 트랙 판정 대상이 아니라 빠져 있기 때문이다.
    """
    ts = pd.Timestamp(asof)
    px_v = px[px["date"] <= ts]
    if px_v.empty:
        return None
    last_px = px_v.sort_values("date").groupby("ticker")["close_adj"].last()

    if isinstance(shares, pd.DataFrame):
        shares = {t: float(v) for t, v in
                  shares["shares_outstanding"].items() if v and v > 0}

    caps, rows = {}, []
    for t, hist in ttm.groupby("ticker"):
        sh = shares.get(t)
        if not sh or t not in last_px.index:
            continue
        vis = hist[pd.to_datetime(hist["rcept_dt"]) <= ts]
        if vis.empty:
            continue
        row = vis.sort_values("period").iloc[-1]
        track = str(tracks.loc[t, "track"]) if t in tracks.index else "T1"
        cap = float(last_px[t]) * sh
        caps[t] = cap
        m = metrics.compute(row.to_dict(), market_cap=cap, shares_out=sh,
                            hist=vis, track=track)
        rows.append({"ticker": t, "name": t, "track": track,
                     "market_cap": cap, **m,
                     # 이 행을 채점한 재무가 **언제 공시된 것인지** 남긴다.
                     # 없으면 3년 묵은 숫자로 채점되고 있어도 알 수가 없다.
                     "rcept_dt": row.get("rcept_dt"),
                     "fin_period": row.get("period")})
    if not rows:
        return None

    df = pd.DataFrame(rows)
    mom = momentum.build(px_v, None, caps, asof)
    df = df.merge(mom, on="ticker", how="left")
    g = grading.build(df, sector)
    g["asof"] = asof
    return g


def month_starts(start: str, end: str, day: int = 5) -> list[str]:
    out, y, m = [], int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}-{day:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out
