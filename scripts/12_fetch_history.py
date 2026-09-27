"""검증용 과거 데이터 수집 — 2018~2022 유니버스 합집합.

    python scripts/12_fetch_history.py --what dart   # 재무 (DART, 일일 2만 회 한도)
    python scripts/12_fetch_history.py --what krx    # 시세 (KRX, 속도 제한 있음)
    python scripts/12_fetch_history.py --what dart --limit 150   # 오늘 여기까지만

**한 번에 하나만 돌린다.** 어제 KRX가 차단된 직접적 원인이 두 작업의 동시 실행이었다.
이미 받은 종목은 캐시를 타므로 중단 후 다시 실행해도 처음부터 받지 않는다.
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

from src import dart, krx, quarterly, store, ttm, universe  # noqa: E402

YEARS = list(range(2017, 2023))          # TTM은 전년이 있어야 하므로 2017부터
PRICE_START, PRICE_END = "2017-01-01", "2023-06-30"


def log(m: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def target_tickers() -> list[str]:
    """한 번이라도 유니버스에 편입된 종목 전부."""
    return universe.included_tickers()


def fetch_dart(tickers: list[str], limit: int) -> int:
    cc = dart.corp_codes()
    cmap = dict(cc[["ticker", "corp_code"]].values)
    have = {p.stem.split("_")[0] for p in
            (dart.config.RAW / "dart" / "fin").glob("*.json")} \
        if (dart.config.RAW / "dart" / "fin").exists() else set()

    todo = [t for t in tickers if cmap.get(t)]
    log(f"대상 {len(todo)}종목 × {len(YEARS)}년 (2017~2022)")
    log(f"예상 호출 약 {len(todo)*len(YEARS)*4*2:,}회 — 일일 한도 20,000회")

    q_all, t_all, done, fails = [], [], 0, []
    t0 = time.monotonic()
    for i, tk in enumerate(todo, 1):
        if limit and done >= limit:
            log(f"--limit {limit} 도달 — 여기서 멈춥니다. 내일 이어서 실행하세요.")
            break
        try:
            q = quarterly.fetch_company(cmap[tk], tk, YEARS)
            if not q.empty and "quarter" in q.columns:
                q_all.append(q)
                t_all.append(ttm.build(q))
            done += 1
        except Exception as exc:
            fails.append((tk, str(exc)[:60]))
        if i % 20 == 0:
            el = time.monotonic() - t0
            log(f"  {i}/{len(todo)}  경과 {el/60:.1f}분  남은 예상 {el/i*(len(todo)-i)/60:.1f}분")

    if q_all:
        store.save(pd.concat(q_all, ignore_index=True), "fin_quarterly_hist")
        store.save(pd.concat(t_all, ignore_index=True), "fin_ttm_hist")
        log(f"저장: fin_ttm_hist {sum(len(x) for x in t_all):,}행 / {len(t_all)}종목")
    if fails:
        log(f"실패 {len(fails)}종목: " + ", ".join(t for t, _ in fails[:10]))
    return done


def fetch_krx(tickers: list[str], limit: int) -> int:
    log(f"대상 {len(tickers)}종목 · {PRICE_START} ~ {PRICE_END}")
    log(f"속도 제한 {krx.MIN_INTERVAL}초/호출 — 예상 {len(tickers)*krx.MIN_INTERVAL/60:.0f}분")
    px, done, fails = [], 0, []
    t0 = time.monotonic()
    for i, tk in enumerate(tickers, 1):
        if limit and done >= limit:
            log(f"--limit {limit} 도달 — 여기서 멈춥니다.")
            break
        try:
            d = krx.ohlcv(tk, PRICE_START, PRICE_END)
            if not d.empty:
                px.append(d)
            done += 1
        except Exception as exc:
            fails.append((tk, str(exc)[:60]))
        if i % 25 == 0:
            el = time.monotonic() - t0
            log(f"  {i}/{len(tickers)}  경과 {el/60:.1f}분  "
                f"남은 예상 {el/i*(len(tickers)-i)/60:.1f}분  KRX {krx.call_count()}회")
        if krx.call_count() >= krx.DAILY_SOFT_LIMIT:
            log("일일 소프트 한도 도달 — 중단합니다. 내일 이어서 실행하세요.")
            break

    if px:
        df = pd.concat(px, ignore_index=True)
        store.save(df, "prices_daily_hist")
        log(f"저장: prices_daily_hist {len(df):,}행 / {df.ticker.nunique()}종목")
    if fails:
        log(f"실패 {len(fails)}종목: " + ", ".join(t for t, _ in fails[:10]))
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["dart", "krx"], required=True)
    ap.add_argument("--limit", type=int, default=0, help="이번 실행에서 처리할 종목 수 상한")
    args = ap.parse_args()

    try:
        tickers = target_tickers()
    except FileNotFoundError:
        log("universe_history_2018_2022 이 없습니다. 10_resume_krx.py 를 먼저 실행하세요.")
        return 1
    log(f"수집 대상 합집합 {len(tickers)}종목")

    n = fetch_dart(tickers, args.limit) if args.what == "dart" \
        else fetch_krx(tickers, args.limit)
    log(f"완료 — {n}종목 처리")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
