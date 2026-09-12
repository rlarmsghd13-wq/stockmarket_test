"""L4 — 세 전략 스코어링 → 전략별 상위 종목.

    python scripts/06_score_strategies.py
    python scripts/06_score_strategies.py --top 5
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src import grading, store, strategies  # noqa: E402

CATS = ["growth", "profitability", "stability", "valuation", "momentum"]
SHORT = {"growth": "성장", "profitability": "수익", "stability": "안정",
         "valuation": "밸류", "momentum": "모멘텀"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    g = store.load("grades_latest")
    ttm = store.load("fin_ttm")
    gf = strategies.gate_features(ttm, args.asof)
    df = g.merge(gf, on="ticker", how="left")

    all_scores = []
    for s in (strategies.VALUE, strategies.GROWTH):
        r = strategies.score(df, s)
        all_scores.append(r)
        passed = r[r.gate_pass]
        print(f"\n{'='*96}")
        print(f"[{strategies.LABEL[s]}]  게이트 통과 {len(passed)}/{len(r)}종목"
              f"   가중치 " + " · ".join(
                  f"{SHORT[c]} {w}" for c, w in strategies.WEIGHTS[s].items() if w))
        print("=" * 96)
        if passed.empty:
            print("  통과 종목 없음")
            top_fail = r.nlargest(3, "base_score")
            print("  (점수 상위 3종목의 탈락 사유)")
            for x in top_fail.itertuples():
                print(f"    {x.name[:14]:<15} {x.base_score:>5.1f}  {x.gate_fails}")
            continue

        head = f"  {'종목':<15}{'점수':>7}{'기본':>7}{'감점':>6}  "
        head += "".join(f"{SHORT[c]:>7}" for c in CATS)
        print(head)
        print("  " + "-" * (len(head) + 8))
        for x in passed.head(args.top).itertuples():
            row = g[g.ticker == x.ticker].iloc[0]
            line = f"  {x.name[:14]:<15}{x.score:>7.1f}{x.base_score:>7.1f}{x.penalty:>6.0f}  "
            for c in CATS:
                v = row[f"{c}_score"]
                line += f"{v:>7.0f}" if pd.notna(v) else f"{'—':>7}"
            print(line)
            if x.penalty_why:
                print(f"    └ {x.penalty_why}")

    res = pd.concat(all_scores, ignore_index=True)
    store.save(res, "strategy_scores")

    print(f"\n{'='*96}")
    print("전략 간 중복 (여러 전략의 게이트를 동시에 통과한 종목):")
    p = res[res.gate_pass]
    dup = p.groupby("ticker").filter(lambda x: len(x) > 1)
    if dup.empty:
        print("  없음 — 세 전략이 서로 다른 종목을 고르고 있다")
    else:
        for t, x in dup.groupby("ticker"):
            best = x.loc[x.score.idxmax()]
            others = ", ".join(f"{strategies.LABEL[r.strategy]} {r.score:.0f}"
                               for r in x.itertuples())
            print(f"  {x.iloc[0]['name'][:14]:<15} {others}"
                  f"  → {strategies.LABEL[best.strategy]}에 귀속")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
