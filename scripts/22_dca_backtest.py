"""적립식 백테스트 — 초기자본 + 매달 정액, 상위 N종목, 손절·분할익절.

    python scripts/22_dca_backtest.py
    python scripts/22_dca_backtest.py --initial 20000000 --contrib 500000 --top 5

17번(매달 상위 5개로 갈아타기)과 달리, `equal`·`fill` 배분에서는
**순위에서 밀려도 팔지 않는다.** 손절 또는 익절에 닿을 때만 판다.
`rebalance` 배분만 순위 이탈 시 매도한다.

증권사 목표가는 과거 데이터가 없어(2026-09부터 수집) **매수가 대비 비율**로
대체한다. 사용자 지정: 1차 +30%에서 절반, 2차 +100%에서 잔량.
"""
from __future__ import annotations

import argparse
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

from src import btdata, dca, store  # noqa: E402

STOPS = [None, 30, 25, 20, 15, 10]
T1, T2 = 30, 100
ALLOCS = [("equal", "균등 신규배분"), ("fill", "비중 보정 매수"), ("rebalance", "완전 리밸런싱")]
SIGNALS = ("성장성 부문", "성장주 게이트O")


def picks_for(gp: pd.DataFrame, signal: str, n_top: int) -> dict[str, list[str]]:
    d = gp[gp["in_universe"]].copy()
    if signal == "성장주 게이트O":
        d = d[(d["strategy"] == "growth") & d["pass"]]
        col = "score"
    elif signal == "성장성 부문":
        d = d[d["strategy"] == "growth"]
        col = "growth_score"
    else:
        raise ValueError(signal)
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d[d[col].notna()]
    return {a: list(g.nlargest(n_top, col)["ticker"]) for a, g in d.groupby("asof")}


def bench(px, gp, asofs, contrib, initial) -> dict:
    """같은 현금흐름을 유니버스 전 종목 균등에 넣었을 때 (팔지 않음)."""
    rets = btdata.monthly_returns(asofs)
    u = gp[gp["in_universe"]][["asof", "ticker"]].drop_duplicates()
    eq = u.merge(rets, on=["asof", "ticker"]).groupby("asof")["ret_m"].mean()
    eq = eq.reindex(asofs).fillna(0.0) / 100
    nav, units, val, navs = 1000.0, 0.0, 0.0, []
    for i, a in enumerate(asofs):
        add = contrib + (initial if i == 0 else 0.0)
        units += add / nav
        val += add
        r = float(eq.loc[a])
        val *= (1 + r)
        nav *= (1 + r)
        navs.append(nav)
    paid = initial + contrib * len(asofs)
    twr = navs[-1] / 1000 - 1
    yrs = len(asofs) / 12
    return {"전략": "[벤치] 유니버스 균등", "납입": paid, "최종": val,
            "납입대비": (val / paid - 1) * 100, "기준가수익": twr * 100,
            "CAGR": ((1 + twr) ** (1 / yrs) - 1) * 100,
            "MDD": dca.mdd(pd.Series(navs)), "매도건수": 0,
            "승률": np.nan, "평균보유": np.nan}


def line(r: dict) -> str:
    def f(v, p=1):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{p}f}"
    return (f"  {r['전략']:<30}{r['최종']/1e4:>11,.0f}만{r['납입대비']:>+9.1f}%"
            f"{r['기준가수익']:>+9.1f}%{r['CAGR']:>7.1f}%{r['MDD']:>8.1f}%"
            f"{r['매도건수']:>6}{f(r['승률']):>6}%{f(r['평균보유']):>7}")


HEAD = (f"  {'조합':<30}{'최종평가':>13}{'납입대비':>10}{'기준가':>10}"
        f"{'CAGR':>7}{'MDD':>8}{'매도':>6}{'승률':>7}{'보유':>7}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial", type=float, default=20_000_000)
    ap.add_argument("--contrib", type=float, default=500_000)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--end", default=None, help="이 기준일까지만 (예: 2024-06)")
    args = ap.parse_args()
    dca.CONTRIB = args.contrib

    gp = btdata.prepare()
    asofs = sorted(gp["asof"].unique())
    if args.end:
        asofs = [a for a in asofs if a[:7] <= args.end]
    px = pd.concat([store.load("prices_daily_hist"), store.load("prices_daily")],
                   ignore_index=True).drop_duplicates(["ticker", "date"], keep="last")
    px["date"] = pd.to_datetime(px["date"])
    paid = args.initial + args.contrib * len(asofs)

    print(f"초기자본 {args.initial/1e4:,.0f}만원 + 매달 {args.contrib/1e4:,.0f}만원"
          f" × {len(asofs)}개월 = 총 투입 {paid/1e4:,.0f}만원")
    print(f"상위 {args.top}종목 · 1차 +{T1}% 절반 · 2차 +{T2}% 잔량 · 편도비용 {dca.COST}%")
    print(f"기간 {asofs[0]} ~ {asofs[-1]}\n")
    print("=" * 112)
    print(HEAD)
    print(line(bench(px, gp, asofs, args.contrib, args.initial)))

    best, detail = {}, {}
    for signal in SIGNALS:
        pk = picks_for(gp, signal, args.top)
        for mode, mlab in ALLOCS:
            rows = []
            for stop in STOPS:
                res = dca.simulate(pk, px, asofs, contrib=args.contrib, n_top=args.top,
                                   initial=args.initial, alloc=mode,
                                   stop=stop, t1=T1, t2=T2)
                r = dca.summarize(res, "손절없음" if stop is None else f"손절 -{stop}%")
                rows.append(r)
                detail[(signal, mlab, stop)] = res
            rows.sort(key=lambda x: -x["최종"])
            print("\n" + "=" * 112)
            print(f"[{signal}] · {mlab}")
            print("=" * 112)
            print(HEAD)
            for r in rows:
                print(line(r))
            best[(signal, mlab)] = rows[0]

    print("\n" + "=" * 112)
    print("배분 방식 비교 — 각 방식의 최고 조합")
    print("=" * 112)
    print(HEAD)
    for (signal, mlab), r in sorted(best.items(), key=lambda kv: -kv[1]["최종"]):
        print(line(dict(r, 전략=f"{signal}·{mlab} ({r['전략']})")))

    print("\n" + "=" * 112)
    print("매도 사유 (손절 -20%)")
    print("=" * 112)
    for (signal, mlab, stop), res in detail.items():
        if stop != 20 or res.trades.empty:
            continue
        t = res.trades
        parts = []
        for reason, g in t.groupby("reason"):
            parts.append(f"{reason} {len(g)}건 {g['ret_pct'].mean():+.1f}%")
        print(f"  {signal}·{mlab:<12} " + " · ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
