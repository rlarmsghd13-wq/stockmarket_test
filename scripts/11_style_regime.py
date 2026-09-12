"""스타일 국면 — 가치와 성장 중 어느 쪽이 먹히고 있는가.

    python scripts/11_style_regime.py --start 2023-01

월별 스냅샷을 만들고, 각 구간 시작 시점의 명단으로 바스켓을 짜서
다음 달까지의 수익률을 잰다. 결과는 style_spread.parquet에 저장되고
볼트 월간 노트에 국면 요약으로 들어간다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src import panel, store, style  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--top-pct", type=float, default=style.TOP_PCT)
    args = ap.parse_args()

    ttm = store.load("fin_ttm")
    px = store.load("prices_daily")
    px["date"] = pd.to_datetime(px["date"])
    tracks = store.load("tracks").set_index("ticker")
    g0 = store.load("grades_latest")
    shares = g0.set_index("ticker")[["shares_outstanding"]]
    sector = {t: str(tracks.loc[t, "induty_code"])[:2] for t in tracks.index}

    # 시세가 있어야 바스켓 수익률을 잴 수 있다. 모멘텀 지표에도 창이 필요하므로
    # 시세 시작에서 1년은 띄운다.
    first = (px["date"].min() + timedelta(days=370)).strftime("%Y-%m")
    start = max(args.start, first)
    dates = panel.month_starts(start, args.asof[:7])
    print(f"시세 {px['date'].min():%Y-%m-%d} ~ {px['date'].max():%Y-%m-%d}")
    print(f"월별 스냅샷 {len(dates)}개: {dates[0]} ~ {dates[-1]}\n")

    snaps, t0 = {}, time.monotonic()
    for i, d in enumerate(dates, 1):
        c = panel.cohort(d, ttm, px, shares, tracks, sector)
        if c is not None:
            snaps[d] = c
        if i % 6 == 0:
            el = time.monotonic() - t0
            print(f"  {i}/{len(dates)}  경과 {el:.0f}초")
    if len(snaps) < 3:
        print("스냅샷이 부족합니다.")
        return 1

    sp = style.spread_series(snaps, px, args.top_pct)
    store.save(sp, "style_spread")

    print(f"\n스프레드 구간 {len(sp)}개 (상위 {args.top_pct*100:.0f}% 바스켓)")
    print(f"  {'구간':<26}{'가치':>9}{'성장':>9}{'스프레드':>11}")
    print("  " + "-" * 55)
    for r in sp.dropna(subset=["spread"]).itertuples():
        print(f"  {r._1} → {r.to:<12}{r.value_ret:>+8.1f}%{r.growth_ret:>+8.1f}%"
              f"{r.spread:>+10.1f}%p")

    d = sp.dropna(subset=["spread"])
    if not d.empty:
        win = (d["spread"] > 0).mean() * 100
        print(f"\n  전 구간 평균 스프레드 {d['spread'].mean():+.2f}%p "
              f"· 가치가 이긴 달 {win:.0f}%")

    for m in (3, 6, 12):
        reg = style.regime(sp, m)
        if reg["spread"] is None:
            continue
        alloc = style.allocation(reg)
        print(f"\n최근 {reg['months']}개월 → {reg['label']} "
              f"(가치 {reg['value_ret']:+.1f}% · 성장 {reg['growth_ret']:+.1f}% "
              f"· 스프레드 {reg['spread']:+.1f}%p)")
        print(f"  배분 제안: 가치 {alloc['value']}% · 성장 {alloc['growth']}% "
              f"· 급등 {alloc['momentum']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
