"""과거 유니버스 시세를 현재까지 연장 — 생존편향 재발 복구.

    python scripts/20_extend_prices.py --dry-run      # 대상만 확인
    python scripts/20_extend_prices.py                # 실제 수집
    python scripts/20_extend_prices.py --limit 100    # 오늘 여기까지만

왜 필요한가
    `prices_daily_hist`는 12_fetch_history.py가 PRICE_END="2023-06-30"으로
    받아둔 것이고, `prices_daily`는 **현재** 상위 200종목만 담고 있다.
    그래서 2023-07 이후 백테스트 표본이 423종목 → 190종목으로 줄었다.
    생존편향을 없애려고 과거 유니버스를 모았는데, 최근 3년 구간에서
    다시 "지금 살아남은 회사만" 보는 상태가 되어 있었다.

    2024~2025 성과가 특히 좋게 나온 구간이 정확히 이 오염 구간이다.

한 번에 하나만 돌린다. 과거 KRX 차단의 직접 원인이 두 작업 동시 실행이었다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src import env, krx, store  # noqa: E402,F401

# 이어붙일 때 겹침이 있어야 수정주가 배율이 어긋났는지 확인할 수 있다
OVERLAP_START = "2023-06-01"


def log(m: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def targets(end: str) -> tuple[list[str], pd.DataFrame]:
    """연장이 필요한 종목 — 마지막 시세가 최근이 아닌 것."""
    hist = store.load("prices_daily_hist")
    hist["date"] = pd.to_datetime(hist["date"])
    cur = store.load("prices_daily")
    cur["date"] = pd.to_datetime(cur["date"])

    last = pd.concat([hist[["ticker", "date"]], cur[["ticker", "date"]]]) \
             .groupby("ticker")["date"].max()
    cutoff = pd.Timestamp(end) - pd.Timedelta(days=20)
    need = sorted(last[last < cutoff].index)
    return need, hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    need, hist = targets(args.end)
    log(f"연장 대상 {len(need)}종목 · {OVERLAP_START} ~ {args.end}")
    log(f"예상 KRX 호출 {len(need)}회 (일일 소프트 한도 {krx.DAILY_SOFT_LIMIT})")
    if args.dry_run:
        print("  " + ", ".join(need[:30]) + (" ..." if len(need) > 30 else ""))
        return 0

    ok, msg = krx.available()
    if not ok:
        log(f"KRX 접근 불가: {msg} — 중단합니다. 나중에 다시 실행하세요.")
        return 1

    got, empty, fails = [], [], []
    t0 = time.monotonic()
    for i, tk in enumerate(need, 1):
        if args.limit and i > args.limit:
            log(f"--limit {args.limit} 도달 — 여기서 멈춥니다.")
            break
        try:
            d = krx.ohlcv(tk, OVERLAP_START, args.end)
            if d.empty:
                empty.append(tk)          # 상장폐지·거래정지 후보
            else:
                got.append(d)
        except Exception as exc:
            fails.append((tk, str(exc)[:60]))
        if i % 25 == 0:
            el = time.monotonic() - t0
            log(f"  {i}/{len(need)}  경과 {el/60:.1f}분  "
                f"남은 예상 {el/i*(len(need)-i)/60:.1f}분  KRX {krx.call_count()}회")
        if krx.call_count() >= krx.DAILY_SOFT_LIMIT:
            log("일일 소프트 한도 도달 — 중단합니다. 내일 이어서 실행하세요.")
            break

    if not got:
        log("받은 데이터가 없습니다.")
        return 1

    new = pd.concat(got, ignore_index=True)
    new["date"] = pd.to_datetime(new["date"])
    keep = [c for c in hist.columns if c in new.columns]
    merged = (pd.concat([hist, new[keep]], ignore_index=True)
                .drop_duplicates(["ticker", "date"], keep="last")
                .sort_values(["ticker", "date"]))
    store.save(merged, "prices_daily_hist")

    last = merged.groupby("ticker")["date"].max()
    log(f"저장: prices_daily_hist {len(merged):,}행 / {merged['ticker'].nunique()}종목")
    log(f"  연장 성공 {new['ticker'].nunique()}종목 · "
        f"빈 응답 {len(empty)}종목 · 실패 {len(fails)}종목")
    log(f"  최종 시세 종료일 분포: 최근 {(last >= pd.Timestamp(args.end) - pd.Timedelta(days=20)).sum()}종목"
        f" / 그 이전 {(last < pd.Timestamp(args.end) - pd.Timedelta(days=20)).sum()}종목")
    if empty:
        log("  빈 응답(상장폐지·거래정지 후보): " + ", ".join(empty[:20]))
    if fails:
        log("  실패: " + ", ".join(t for t, _ in fails[:10]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
