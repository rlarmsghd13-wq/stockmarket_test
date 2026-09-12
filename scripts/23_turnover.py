"""종목 순환율 — 얼마나 자주 바뀌고, 얼마나 오래 들고 있나.

    python scripts/23_turnover.py

두 가지를 나눠서 잰다.
  ① 신호 순환율 — 매달 상위 N종목 중 몇 개가 교체되는가 (매매 규칙과 무관)
  ② 매매 회전율 — 실제로 얼마를 사고팔았는가, 평균 보유기간, 비용 부담
"""
from __future__ import annotations

import argparse
import importlib.util
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

_spec = importlib.util.spec_from_file_location(
    "d22", os.path.join(os.path.dirname(os.path.abspath(__file__)), "22_dca_backtest.py"))
d22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d22)


def signal_turnover(pk: dict[str, list[str]], asofs: list[str], n: int) -> dict:
    """매달 상위 n종목 중 몇 개가 새로 들어왔나."""
    keep, new, names = [], [], set()
    prev = None
    for a in asofs:
        cur = set(pk.get(a, [])[:n])
        names |= cur
        if prev is not None and cur:
            k = len(cur & prev)
            keep.append(k / len(cur))
            new.append(len(cur - prev))
        prev = cur if cur else prev
    return {"유지비율": float(np.mean(keep)) * 100, "월평균 신규": float(np.mean(new)),
            "연간 교체": float(np.mean(new)) * 12, "등장 종목수": len(names)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--initial", type=float, default=20_000_000)
    ap.add_argument("--contrib", type=float, default=500_000)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    dca.CONTRIB = args.contrib

    gp = btdata.prepare()
    asofs = sorted(gp["asof"].unique())
    px = pd.concat([store.load("prices_daily_hist"), store.load("prices_daily")],
                   ignore_index=True).drop_duplicates(["ticker", "date"], keep="last")
    px["date"] = pd.to_datetime(px["date"])
    yrs = len(asofs) / 12

    print(f"기간 {asofs[0]} ~ {asofs[-1]} ({len(asofs)}개월) · 상위 {args.top}종목\n")
    print("=" * 96)
    print("① 신호 순환율 — 매매 규칙과 무관하게, 상위 5종목 목록이 얼마나 바뀌나")
    print("=" * 96)
    print(f"  {'신호':<16}{'전월 유지':>10}{'월 신규편입':>12}{'연 교체':>9}{'8년간 등장':>11}")
    picks = {}
    for sig in d22.SIGNALS:
        pk = d22.picks_for(gp, sig, args.top)
        picks[sig] = pk
        t = signal_turnover(pk, asofs, args.top)
        print(f"  {sig:<16}{t['유지비율']:>9.0f}%{t['월평균 신규']:>11.1f}개"
              f"{t['연간 교체']:>8.0f}개{t['등장 종목수']:>10}종목")

    print("\n" + "=" * 96)
    print("② 매매 회전율 — 실제 사고판 금액 기준")
    print("=" * 96)
    print(f"  {'조합':<30}{'연 회전율':>10}{'평균보유':>10}{'연 매수':>8}{'연 매도':>8}"
          f"{'비용부담':>10}{'보유종목':>9}")
    for sig in d22.SIGNALS:
        for mode, mlab in d22.ALLOCS:
            res = dca.simulate(picks[sig], px, asofs, contrib=args.contrib,
                               n_top=args.top, initial=args.initial, alloc=mode,
                               stop=20, t1=d22.T1, t2=d22.T2)
            free = dca.simulate(picks[sig], px, asofs, contrib=args.contrib,
                                n_top=args.top, initial=args.initial, alloc=mode,
                                stop=20, t1=d22.T1, t2=d22.T2, cost=0.0)
            avg_val = res.value.mean()
            sold = res.trades["amount"].sum() if len(res.trades) else 0.0
            bought = res.buys["amount"].sum() if len(res.buys) else 0.0
            # 회전율 = 연간 매도금액 / 평균 평가액 (100%면 1년에 한 번 전부 교체)
            turn = sold / yrs / avg_val * 100
            hd = res.trades["hold_days"].dropna() if len(res.trades) else pd.Series(dtype=float)
            drag = (free.final / res.final - 1) * 100
            print(f"  {sig + '·' + mlab:<30}{turn:>9.0f}%"
                  f"{(hd.mean()/30.4 if len(hd) else np.nan):>9.1f}달"
                  f"{len(res.buys)/yrs:>7.0f}건{len(res.trades)/yrs:>7.0f}건"
                  f"{drag:>9.1f}%{res.holdings.mean():>8.1f}개")

    print("\n  회전율 100% = 1년에 보유자산 전부를 한 번 교체한다는 뜻")
    print("  비용부담 = 거래비용을 0으로 뒀을 때 최종평가액이 얼마나 더 커지는가")

    print("\n" + "=" * 96)
    print("③ 매도 사유별 보유기간 (성장성 부문)")
    print("=" * 96)
    for mode, mlab in d22.ALLOCS:
        res = dca.simulate(picks["성장성 부문"], px, asofs, contrib=args.contrib,
                           n_top=args.top, initial=args.initial, alloc=mode,
                           stop=20, t1=d22.T1, t2=d22.T2)
        t = res.trades
        if t.empty:
            continue
        parts = []
        for reason, g in t.groupby("reason"):
            d = g["hold_days"].dropna()
            parts.append(f"{reason} {len(g)}건 {d.mean()/30.4:.1f}달" if len(d)
                         else f"{reason} {len(g)}건")
        print(f"  {mlab:<14} " + " · ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
