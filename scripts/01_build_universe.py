"""L0 — 유니버스 스냅샷 생성.

    python scripts/01_build_universe.py            # 오늘 시점만
    python scripts/01_build_universe.py --history  # 2015년부터 리밸런싱 시점 전부

--history는 백테스트용이다. 각 시점에 실제로 상장돼 있던 종목으로 유니버스를
다시 구성하므로, 이후 상장폐지된 종목이 자연스럽게 포함된다 (생존편향 제거).
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

import config                                  # noqa: E402
from src import env, krx, store, universe      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", action="store_true",
                    help="2015년부터 모든 리밸런싱 시점")
    ap.add_argument("--asof", default=date.today().isoformat())
    args = ap.parse_args()

    print("환경변수:", env.status())

    if args.history:
        dates = krx.rebalance_dates()
        print(f"리밸런싱 시점 {len(dates)}개: {dates[0]} ~ {dates[-1]}")
        df = universe.build_history(dates)
        # 이미 만들어둔 스냅샷을 지우지 않는다. rebalance_dates()는
        # config.BACKTEST_START 이후만 주므로, 덮어쓰면 그 앞 구간이 사라진다.
        if store.exists("universe_history"):
            df = (pd.concat([store.load("universe_history"), df], ignore_index=True)
                    .drop_duplicates(["ticker", "asof_date"], keep="last"))
        name = "universe_history"
    else:
        df = universe.snapshot(args.asof)
        name = f"universe_{args.asof.replace('-', '')}"

    path = store.save(df, name)
    inc = df[df["included"]]
    print(f"\n저장: {path}")
    print(f"전체 {len(df):,}행 / 편입 {len(inc):,}행 / 시점 {df['asof_date'].nunique()}개")
    print("\n제외 사유:")
    print(df["excluded_reason"].value_counts(dropna=False).to_string())

    if not args.history:
        print(f"\n시총 상위 10 (컷: {config.MIN_MARKET_CAP/1e8:,.0f}억):")
        top = inc.sort_values("market_cap", ascending=False).head(10)
        for r in top.itertuples():
            print(f"  {r.ticker} {r.name:<14} {r.market_cap/1e12:>8.1f}조")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
