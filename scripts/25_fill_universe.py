"""유니버스 스냅샷 구멍 메우기 — 2020-11 · 2023~2026.

    python scripts/25_fill_universe.py --dry-run    # 무엇이 빠졌는지만 본다
    python scripts/25_fill_universe.py              # 빠진 스냅샷만 수집
    python scripts/25_fill_universe.py --limit 5    # 오늘은 5개만

왜 필요한가
    스냅샷이 2018-04~2022-11(19개)과 2026-09-05(1개)뿐이라, **2023~2025년에만
    시총 상위 200이었다가 밀려난 종목이 후보 풀에서 통째로 빠져 있었다.**
    실측 누락률은 2023년 6% → 2025년 11%. 에코프로머티·두산로보틱스·테크윙처럼
    급등 후 탈락한 종목이 전략에도 벤치마크에도 없었다는 뜻이다.
    없는 종목은 살 수도 없지만 벤치마크에도 안 들어가므로, 어느 쪽으로
    틀렸는지는 고쳐봐야 안다.

주의
    KRX는 하루 800콜 소프트 한도 · 동시 실행 금지(과거 24시간 차단됨).
    한 스냅샷당 약 20콜이다. **스냅샷 하나를 끝낼 때마다 저장**하므로
    중단하거나 한도에 걸려도 다음 실행이 이어받는다.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src import krx, store, universe  # noqa: E402

# 각 보고서 제출 마감 직후 — 그 시점에 실제로 볼 수 있었던 재무로 판단하게 된다.
REBALANCE = ((4, 5), (5, 20), (8, 20), (11, 20))
START_YEAR = 2018
STORE = "universe_history"
CALLS_PER_SNAPSHOT = 22      # 실측 근사. 한도 계산에만 쓴다


def log(m: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def wanted() -> list[str]:
    today = date.today()
    out = []
    for y in range(START_YEAR, today.year + 1):
        for mm, dd in REBALANCE:
            d = date(y, mm, dd)
            if d <= today:
                out.append(d.isoformat())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="이번 실행에서 만들 스냅샷 수 상한")
    args = ap.parse_args()

    try:
        have = universe.load_snapshots()
    except FileNotFoundError:
        have = pd.DataFrame(columns=["ticker", "asof_date", "included"])
    have_dates = set(have["asof_date"]) if len(have) else set()
    before = set(have[have["included"]]["ticker"]) if len(have) else set()

    need = [d for d in wanted() if d not in have_dates]
    print(f"기준일 {len(wanted())}개 중 보유 {len(have_dates)}개 · "
          f"누락 {len(need)}개")
    print(f"현재 편입 고유종목 {len(before)}개\n")
    if not need:
        print("구멍이 없습니다.")
        return 0
    for d in need:
        print(f"  누락 {d}")
    print(f"\n예상 KRX 호출 약 {len(need)*CALLS_PER_SNAPSHOT:,}회 "
          f"(소프트 한도 {krx.DAILY_SOFT_LIMIT}회/일)")
    if args.dry_run:
        print("\n--dry-run 이라 수집하지 않습니다.")
        return 0

    # KRX가 지금 응답하는지 먼저 1회만 확인한다. 차단 상태에서 재시도하면
    # 차단이 길어질 뿐이다.
    ok, msg = krx.available()
    log(f"KRX 상태: {msg}")
    if not ok:
        print("\n차단 중입니다. 아무것도 수집하지 않고 종료합니다.")
        return 2

    existing = store.load(STORE) if store.exists(STORE) else pd.DataFrame()
    done, fails = 0, []
    for i, d in enumerate(need, 1):
        if args.limit and done >= args.limit:
            log(f"--limit {args.limit} 도달 — 중단합니다.")
            break
        if krx.call_count() + CALLS_PER_SNAPSHOT > krx.DAILY_SOFT_LIMIT:
            log(f"소프트 한도 근접(현재 {krx.call_count()}회) — 중단합니다. "
                "내일 다시 실행하면 이어받습니다.")
            break
        try:
            snap = universe.snapshot(d)
        except Exception as exc:
            log(f"  [실패] {d}: {type(exc).__name__}: {str(exc)[:80]}")
            fails.append(d)
            continue
        existing = pd.concat([existing, snap], ignore_index=True)
        existing = existing.drop_duplicates(["ticker", "asof_date"], keep="last")
        store.save(existing, STORE)          # 스냅샷마다 저장 — 중단에 안전하다
        inc = snap[snap["included"]]
        log(f"  [{i}/{len(need)}] {d}  전체 {len(snap):,}종목 · 편입 {len(inc)}개 · "
            f"KRX {krx.call_count()}회")
        done += 1

    if not done:
        print("\n새로 만든 스냅샷이 없습니다.")
        return 1

    after_all = universe.load_snapshots()
    after = set(after_all[after_all["included"]]["ticker"])
    new = sorted(after - before)
    print(f"\n{'='*74}")
    print(f"스냅샷 {done}개 추가 · KRX 호출 {krx.call_count()}회")
    print(f"편입 고유종목 {len(before)} → {len(after)}  (+{len(new)})")
    print("=" * 74)
    if new:
        nm = (after_all[after_all["ticker"].isin(new)]
              .sort_values("asof_date").groupby("ticker")["name"].last())
        first = (after_all[after_all["included"] & after_all["ticker"].isin(new)]
                 .groupby("ticker")["asof_date"].min())
        print(f"\n새로 발견된 종목 {len(new)}개 — 지금까지 후보에 없던 종목이다")
        print(f"  {'티커':<9}{'종목명':<22}{'첫 편입':<12}")
        for t in sorted(new, key=lambda x: first.get(x, "")):
            print(f"  {t:<9}{str(nm.get(t, '')):<22}{first.get(t, ''):<12}")
    if fails:
        print(f"\n실패 {len(fails)}개: {', '.join(fails)}")

    print("\n다음 단계 (사람이 순서대로 시작한다 — 동시에 돌리지 말 것)")
    print("  1) python scripts/26_fill_new_tickers.py --what dart")
    print("  2) python scripts/26_fill_new_tickers.py --what krx")
    print("  3) python scripts/09_validate.py  →  14_validate_gates.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
