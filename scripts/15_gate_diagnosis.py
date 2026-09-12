"""게이트 정밀진단 — 14번에서 가치 게이트가 수익률을 깎는 것이 확인된 뒤의 후속.

    python scripts/15_gate_diagnosis.py

세 가지를 확인한다.
    ① 연도별 분해 — 게이트의 실패가 전 기간인가, 특정 시기인가
    ② 시가총액 효과 — 게이트를 끄면 들어오는 종목이 무엇인가
    ③ 실행가능성 — 게이트를 끈 초과수익이 "못 사는 종목"에서 나온 것인가

③은 외부 백테스트 검토(_참고/타인백테스트_검토.md)의 4단계 D에서 가져왔다.
그 전략은 상한가 종목을 빼자 수익이 통째로 사라졌다. 같은 칼을 우리에게 댄다.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src import store, strategies  # noqa: E402

TOP_N = 5
LIQ_STEPS = [0, 10e8, 30e8, 50e8, 100e8, 300e8]


def cluster_t(g: pd.DataFrame, col: str = "excess") -> tuple[float, float, int]:
    m = g.groupby("asof")[col].mean().dropna()
    if len(m) < 3 or m.std(ddof=1) == 0:
        return float("nan"), float("nan"), len(m)
    t = m.mean() / (m.std(ddof=1) / math.sqrt(len(m)))
    return float(t), float(math.erfc(abs(t) / math.sqrt(2))), len(m)


def turnover_asof(panel: pd.DataFrame) -> pd.DataFrame:
    """기준일 시점의 60일 평균 거래대금(근사).

    close_adj × volume 이므로 액면분할 구간에서 왜곡된다. 절대값이 아니라
    종목 간 유동성 순위를 보는 용도라 감수한다.
    """
    px = pd.concat([store.load("prices_daily_hist"), store.load("prices_daily")],
                   ignore_index=True).drop_duplicates(["ticker", "date"], keep="last")
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])
    px["turnover"] = px["close_adj"] * px["volume"]
    px["avg_tv"] = (px.groupby("ticker")["turnover"]
                      .transform(lambda s: s.rolling(60, min_periods=20).mean()))
    return px[["ticker", "date", "avg_tv"]].dropna()


def picks(d: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    return d.sort_values("score", ascending=False).groupby("asof").head(n)


def line(nm: str, p: pd.DataFrame, width: int = 20) -> str:
    t, pv, n = cluster_t(p)
    return (f"  {nm:<{width}}{n:>5}{p['fwd_63'].mean():>9.2f}%"
            f"{p['excess'].mean():>9.2f}%"
            f"{(p['fwd_63'] > 0).mean() * 100:>7.1f}%{t:>7.2f}{pv:>8.3f}")


HEAD = (f"  {'':<20}{'월수':>5}{'평균':>10}{'초과':>10}{'승률':>8}{'t':>7}{'p':>8}")


def main() -> int:
    gp = store.load("gate_panel")
    tv = turnover_asof(gp)
    gp["asof_ts"] = pd.to_datetime(gp["asof"])
    gp = pd.merge_asof(gp.sort_values("asof_ts"), tv.sort_values("date"),
                       left_on="asof_ts", right_on="date", by="ticker",
                       direction="backward")
    gp["year"] = gp["asof"].str[:4]
    print(f"패널 {len(gp):,}행 · 거래대금 결측 {gp['avg_tv'].isna().mean() * 100:.1f}%")

    for strat in (strategies.VALUE, strategies.GROWTH):
        d = gp[gp["strategy"] == strat].copy()
        lab = strategies.LABEL[strat]

        # ── ① 연도별 분해 ────────────────────────────────────────────
        print("\n" + "=" * 84)
        print(f"① [{lab}] 연도별 — 게이트 켬 vs 끔 (상위 {TOP_N}종목 초과수익)")
        print("=" * 84)
        print(f"  {'연도':<8}{'게이트 켬':>12}{'게이트 끔':>12}{'차이':>10}"
              f"{'통과종목/월':>13}")
        for y, g in d.groupby("year"):
            on = picks(g[g["pass"]])["excess"].mean()
            off = picks(g)["excess"].mean()
            npass = g.groupby("asof")["pass"].sum().mean()
            diff = on - off if pd.notna(on) else float("nan")
            print(f"  {y:<8}{on:>11.2f}%{off:>11.2f}%{diff:>+9.2f}%{npass:>12.1f}")

        # ── ② 시가총액 효과 ─────────────────────────────────────────
        print("\n" + "=" * 84)
        print(f"② [{lab}] 시가총액 5분위별 초과수익 (전 종목)")
        print("=" * 84)
        d["cap_q"] = d.groupby("asof")["market_cap"].transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
        print(f"  {'분위':<8}{'중앙시총':>12}{'초과수익':>11}{'승률':>9}{'표본':>9}")
        for q, g in d.groupby("cap_q"):
            print(f"  Q{int(q)}{'(소형)' if q == 1 else '(대형)' if q == 5 else '':<5}"
                  f"{g['market_cap'].median() / 1e8:>10.0f}억"
                  f"{g['excess'].mean():>10.2f}%"
                  f"{(g['fwd_63'] > 0).mean() * 100:>8.1f}%{len(g):>9,}")

        # ── ③ 실행가능성 ────────────────────────────────────────────
        print("\n" + "=" * 84)
        print(f"③ [{lab}] 거래대금 하한을 올리면 — 게이트 끈 상위 {TOP_N}종목")
        print("=" * 84)
        print(HEAD)
        for lo in LIQ_STEPS:
            sub = d[d["avg_tv"].fillna(0) >= lo]
            if sub.empty:
                continue
            nm = "제한 없음" if lo == 0 else f"{lo / 1e8:.0f}억 이상"
            print(line(nm, picks(sub)))

        print(f"\n  (참고) 게이트 켠 상위 {TOP_N}종목")
        print(line("게이트 켬", picks(d[d["pass"]])))
        m = d["pass"] | (d["market_cap"] >= 0)     # placeholder for readability
        del m

    # ── ④ 성장주 통과율 = 레짐 지표 검정 ────────────────────────────
    print("\n" + "=" * 84)
    print("④ 게이트 통과율이 이후 시장수익률을 예측하는가")
    print("=" * 84)
    for strat in (strategies.VALUE, strategies.GROWTH):
        d = gp[gp["strategy"] == strat]
        lab = strategies.LABEL[strat]
        c = d.groupby("asof").agg(rate=("pass", "mean"), mkt=("mkt", "first")).dropna()
        c["rate"] *= 100
        r = c["rate"].rank().corr(c["mkt"].rank())
        n = len(c)
        t = r * math.sqrt((n - 2) / max(1e-9, 1 - r * r))
        p = math.erfc(abs(t) / math.sqrt(2))
        print(f"\n[{lab}]  순위상관 {r:+.3f}  t={t:.2f}  p={p:.3f}  (월 {n})")
        q = c["rate"].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
        edges = [-1] + q + [999]
        print(f"  {'통과율 구간':<16}{'월수':>6}{'이후 3개월 시장':>16}")
        for i in range(5):
            s = c[(c["rate"] > edges[i]) & (c["rate"] <= edges[i + 1])]
            if s.empty:
                continue
            print(f"  {s['rate'].min():>5.1f}~{s['rate'].max():<9.1f}"
                  f"{len(s):>6}{s['mkt'].mean():>15.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
