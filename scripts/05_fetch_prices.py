"""시세·수급 적재 — 한 번 받아두고 어떤 기준일이든 로컬에서 잘라 쓴다.

    python scripts/05_fetch_prices.py           # 최근 2년
    python scripts/05_fetch_prices.py --years 3

1~2주 주기로 관심종목을 갱신하려면 매번 KRX를 다시 긁으면 안 된다.
넉넉한 창을 한 번 받아 저장하고, 기준일마다 슬라이스해서 지표를 계산한다.
격주 갱신의 KRX 호출은 0회가 된다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src import krx, store        # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="universe_20260905")
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    end = date.today()
    start = end - timedelta(days=int(365.25 * args.years) + 30)
    log(f"수집 구간 {start} ~ {end}")

    uni = store.load(args.universe)
    inc = uni[uni["included"]].sort_values("market_cap", ascending=False)
    if args.n:
        inc = inc.head(args.n)

    px, fl, fails = [], [], []
    t0 = time.monotonic()
    for i, u in enumerate(inc.itertuples(), 1):
        try:
            d = krx.ohlcv(u.ticker, start.isoformat(), end.isoformat())
            if not d.empty:
                px.append(d)
        except Exception as exc:
            fails.append((u.ticker, u.name, f"ohlcv: {exc}"))
        try:
            f = krx.investor_flows(u.ticker, start.isoformat(), end.isoformat())
            if not f.empty:
                fl.append(f)
        except Exception as exc:
            fails.append((u.ticker, u.name, f"flows: {exc}"))
        if i % 20 == 0:
            el = time.monotonic() - t0
            log(f"  {i}/{len(inc)}  경과 {el/60:.1f}분  남은 예상 {el/i*(len(inc)-i)/60:.1f}분")

    if not px:
        log("시세를 하나도 받지 못했습니다.")
        return 1

    dfp = pd.concat(px, ignore_index=True)
    store.save(dfp, "prices_daily")
    log(f"prices_daily {len(dfp):,}행 / {dfp.ticker.nunique()}종목")

    if fl:
        dff = pd.concat(fl, ignore_index=True)
        store.save(dff, "flows_daily")
        log(f"flows_daily  {len(dff):,}행 / {dff.ticker.nunique()}종목")
    else:
        log("투자자별 순매수를 받지 못했습니다 — 수급 지표는 NULL이 됩니다.")

    log(f"소요 {(time.monotonic()-t0)/60:.1f}분")
    if fails:
        log(f"실패 {len(fails)}건:")
        for t, n, why in fails[:20]:
            log(f"  {t} {n}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
