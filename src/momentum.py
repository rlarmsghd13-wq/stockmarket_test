"""L2 — 모멘텀·수급 지표.

부문 다섯 개 중 **1~2주 주기로 실제로 바뀌는 유일한 부문**이다.
재무제표는 분기에 한 번만 갱신되므로, 격주 리포트에서 순위를 움직이는 건
여기와 주가에 연동된 밸류에이션뿐이다.

저장된 시계열을 기준일로 잘라서 계산한다 — KRX를 다시 호출하지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 거래일 기준 창
W_1M, W_3M, W_6M, W_52W = 21, 63, 126, 252
W_VOL_SHORT, W_VOL_LONG = 5, 60
W_FLOW, W_MA, W_RSI = 20, 20, 14


def _ret(close: pd.Series, n: int):
    if len(close) <= n:
        return None
    prev = close.iloc[-(n + 1)]
    if pd.isna(prev) or prev <= 0:
        return None
    return (close.iloc[-1] / prev - 1) * 100


def _rsi(close: pd.Series, n: int = W_RSI):
    if len(close) < n + 1:
        return None
    d = close.diff().dropna()
    if len(d) < n:
        return None
    up = d.clip(lower=0).rolling(n).mean().iloc[-1]
    dn = (-d.clip(upper=0)).rolling(n).mean().iloc[-1]
    if pd.isna(up) or pd.isna(dn):
        return None
    if dn == 0:
        return 100.0
    rs = up / dn
    return 100 - 100 / (1 + rs)


def compute(prices: pd.DataFrame, flows: pd.DataFrame | None,
            asof: str, market_cap: float | None) -> dict:
    """한 종목의 기준일 시점 모멘텀·수급 지표.

    prices: 그 종목의 일봉 (date, close_adj, volume, trading_value)
    flows : 그 종목의 투자자별 순매수 (date, foreign_net, inst_net)
    """
    out: dict[str, float | None] = {
        "return_1m": None, "return_3m": None, "return_6m": None,
        "volume_ratio": None, "pos_52w": None, "disparity_20": None,
        "rsi_14": None, "price_asof": None, "price_days": 0,
    }
    if prices is None or prices.empty:
        return out

    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["date"] <= pd.Timestamp(asof)].sort_values("date")
    if p.empty:
        return out
    out["price_days"] = len(p)

    close = p["close_adj"].astype(float).reset_index(drop=True)
    vol = p["volume"].astype(float).reset_index(drop=True)
    out["price_asof"] = float(close.iloc[-1])

    out["return_1m"] = _ret(close, W_1M)
    out["return_3m"] = _ret(close, W_3M)
    out["return_6m"] = _ret(close, W_6M)

    if len(vol) >= W_VOL_LONG:
        short = vol.iloc[-W_VOL_SHORT:].mean()
        long = vol.iloc[-W_VOL_LONG:].mean()
        out["volume_ratio"] = float(short / long) if long > 0 else None

    if len(close) >= W_52W:
        hi = close.iloc[-W_52W:].max()
        out["pos_52w"] = float(close.iloc[-1] / hi * 100) if hi > 0 else None

    if len(close) >= W_MA:
        ma = close.iloc[-W_MA:].mean()
        # 20일 이격도 — 과열 필터. 130% 이상이면 추격매수 구간으로 본다.
        out["disparity_20"] = float(close.iloc[-1] / ma * 100) if ma > 0 else None

    out["rsi_14"] = _rsi(close)

    # 투자자별 순매수는 쓰지 않는다 — KRX가 긴 구간 조회를 거부해
    # 종목별 결측이 크고, 결측이 큰 지표는 종목마다 다른 잣대를 만든다.
    return out


def build(prices: pd.DataFrame, flows: pd.DataFrame | None,
          caps: dict[str, float], asof: str) -> pd.DataFrame:
    """전 종목 일괄."""
    rows = []
    fl_by = ({t: g for t, g in flows.groupby("ticker")}
             if flows is not None and not flows.empty else {})
    for t, g in prices.groupby("ticker"):
        rows.append({"ticker": t,
                     **compute(g, fl_by.get(t), asof, caps.get(t))})
    return pd.DataFrame(rows)


def overheated(m: dict) -> list[str]:
    """과열 경고 — 급등 전략에서 신규 매수를 막는 조건.

    "이미 많이 오른 종목을 추격 매수하지 않는다"를 규칙으로 옮긴 것.
    """
    warn = []
    d = m.get("disparity_20")
    r = m.get("rsi_14")
    p = m.get("pos_52w")
    if d is not None and d >= 130:
        warn.append(f"20일 이격도 {d:.0f}%")
    if r is not None and r >= 80:
        warn.append(f"RSI {r:.0f}")
    if p is not None and p >= 99 and r is not None and r >= 75:
        warn.append("52주 신고가 + 과열")
    return warn
