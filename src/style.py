"""스타일 스프레드 — 지금 시장에서 가치와 성장 중 어느 쪽이 먹히고 있는가.

매크로 이론을 가정하지 않는다. "금리가 오르면 가치주가 좋다" 같은 통념은
한국 시장에서 성립하는지 확인되지 않았고, 투신/사모 백테스트에서 통념이
정반대로 뒤집힌 사례를 이미 봤다. 그래서 **우리 유니버스 안에서 두 스타일이
실제로 어떻게 움직였는지**만 잰다.

    가치 바스켓 = 그 시점 밸류에이션 점수 상위 N%
    성장 바스켓 = 그 시점 성장성 점수 상위 N%
    스프레드   = 가치 수익률 − 성장 수익률

**바스켓은 구간 시작 시점의 명단으로 만든다.** 오늘의 가치주 명단으로 과거
수익률을 재면, 떨어졌기 때문에 싸진 종목이 섞여 결과가 뒤집힌다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOP_PCT = 0.20


def baskets(grades: pd.DataFrame, top_pct: float = TOP_PCT) -> dict[str, list[str]]:
    """그 시점 점수로 두 바스켓을 만든다. 두 명단이 겹칠 수 있다 — 정상이다."""
    out: dict[str, list[str]] = {"value": [], "growth": []}
    n = max(5, int(len(grades) * top_pct))
    for key, col in (("value", "valuation_score"), ("growth", "growth_score")):
        if col not in grades.columns:
            continue
        d = grades[["ticker", col]].dropna()
        if len(d) < 10:
            continue
        out[key] = d.nlargest(n, col)["ticker"].tolist()
    return out


def basket_return(px: pd.DataFrame, tickers: list[str],
                  t0: str, t1: str) -> float | None:
    """동일가중 수익률(%). 구간 안에 시세가 끊긴 종목은 제외한다."""
    if not tickers:
        return None
    a, b = pd.Timestamp(t0), pd.Timestamp(t1)
    d = px[(px["ticker"].isin(tickers)) & (px["date"] >= a) & (px["date"] <= b)]
    if d.empty:
        return None
    rets = []
    for t, g in d.groupby("ticker"):
        g = g.sort_values("date")
        if len(g) < 5:
            continue
        p0, p1 = float(g["close_adj"].iloc[0]), float(g["close_adj"].iloc[-1])
        if p0 > 0:
            rets.append(p1 / p0 - 1)
    return float(np.mean(rets) * 100) if rets else None


def spread_series(snapshots: dict[str, pd.DataFrame], px: pd.DataFrame,
                  top_pct: float = TOP_PCT) -> pd.DataFrame:
    """월별 스냅샷 → 다음 달까지의 스타일 수익률과 스프레드."""
    dates = sorted(snapshots)
    rows = []
    for t0, t1 in zip(dates, dates[1:]):
        b = baskets(snapshots[t0], top_pct)
        v = basket_return(px, b["value"], t0, t1)
        g = basket_return(px, b["growth"], t0, t1)
        rows.append({
            "from": t0, "to": t1,
            "value_ret": v, "growth_ret": g,
            "spread": (v - g) if (v is not None and g is not None) else None,
            "n_value": len(b["value"]), "n_growth": len(b["growth"]),
        })
    return pd.DataFrame(rows)


def regime(spreads: pd.DataFrame, months: int = 3) -> dict:
    """최근 N개월 누적 스프레드로 국면을 판정한다.

    임계값 ±3%p는 노이즈와 신호를 가르는 선일 뿐이며 검증된 값이 아니다.
    검증 전까지는 '무엇이 일어났는지'를 알리는 지표로만 쓴다.
    """
    d = spreads.dropna(subset=["spread"]).tail(months)
    if d.empty:
        return {"label": "판정 불가", "spread": None, "months": 0,
                "value_ret": None, "growth_ret": None, "tilt": None}
    s = float(d["spread"].sum())
    v = float(d["value_ret"].sum())
    g = float(d["growth_ret"].sum())
    if s >= 3:
        label, tilt = "가치 우위", "value"
    elif s <= -3:
        label, tilt = "성장 우위", "growth"
    else:
        label, tilt = "중립", None
    return {"label": label, "spread": s, "months": int(len(d)),
            "value_ret": v, "growth_ret": g, "tilt": tilt}


# 기본 전략 배분. 국면에 따라 여기서만 기울인다 —
# 종목 선정 규칙(게이트·가중치)은 건드리지 않는다. 틀렸을 때 되돌리기 쉽게.
BASE_ALLOC = {"value": 50, "growth": 35, "momentum": 15}
TILT_PT = 10


def allocation(reg: dict) -> dict[str, int]:
    a = dict(BASE_ALLOC)
    if reg.get("tilt") == "value":
        a["value"] += TILT_PT
        a["growth"] -= TILT_PT
    elif reg.get("tilt") == "growth":
        a["growth"] += TILT_PT
        a["value"] -= TILT_PT
    return a


def markdown(reg: dict, alloc: dict, spreads: pd.DataFrame) -> str:
    if reg["spread"] is None:
        return "_스타일 스프레드를 계산할 표본이 부족합니다._"
    lines = [
        f"**{reg['label']}** — 최근 {reg['months']}개월 누적 "
        f"가치 {reg['value_ret']:+.1f}% · 성장 {reg['growth_ret']:+.1f}% "
        f"(스프레드 {reg['spread']:+.1f}%p)",
        "",
        f"전략 배분 제안 — 가치 **{alloc['value']}%** · 성장 **{alloc['growth']}%** "
        f"· 급등 **{alloc['momentum']}%**"
        + ("" if reg["tilt"] else "  (기본값 유지)"),
        "",
        "| 구간 | 가치 | 성장 | 스프레드 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in spreads.dropna(subset=["spread"]).tail(6).itertuples():
        lines.append(f"| {r._1} → {r.to} | {r.value_ret:+.1f}% | "
                     f"{r.growth_ret:+.1f}% | {r.spread:+.1f}%p |")
    lines.append("")
    lines.append("> [!note] 읽는 법")
    lines.append("> 각 구간의 바스켓은 **그 구간 시작 시점의 점수**로 만듭니다. "
                 "오늘 명단으로 과거를 재면 떨어져서 싸진 종목이 섞여 결과가 뒤집힙니다.")
    lines.append("> 배분만 기울이고 **종목 선정 규칙은 바꾸지 않습니다.** "
                 "±3%p 임계값은 아직 검증 전의 값입니다.")
    return "\n".join(lines)
