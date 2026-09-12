"""KRX 차단 해제 확인 후 막혔던 작업 이어가기.

    python scripts/10_resume_krx.py

2026-09-06에 호출을 너무 빠르게·동시에 해서 KRX가 이 IP를 차단했다.
이 스크립트는 그 실수를 반복하지 않도록 만들어졌다.

  · 먼저 **1회만** 호출해 상태를 확인한다. 막혀 있으면 즉시 멈춘다 —
    차단 상태에서 재시도하면 차단이 길어질 뿐이다.
  · 풀렸으면 **과거 유니버스 복원만** 마저 끝낸다 (2018~2022, 남은 스냅샷).
    종목 시세·재무 대량 수집은 여기서 자동 실행하지 않는다. 일일 한도를
    쓰는 작업이라 사람이 보고 시작할 일이다.
  · 모든 호출은 src/krx.py의 속도 제한(초당 약 3회)을 거친다.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src import krx, store, universe  # noqa: E402


def hist_dates() -> list[str]:
    out = []
    for y in range(2018, 2023):
        for mm, dd in ((4, 5), (5, 20), (8, 20), (11, 20)):
            out.append(date(y, mm, dd).isoformat())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="상태만 보고 아무것도 수집하지 않는다")
    args = ap.parse_args()

    print(f"[{datetime.now():%Y-%m-%d %H:%M}] KRX 상태 확인 (호출 1회)")
    ok, msg = krx.available()
    print(f"  → {msg}")

    if not ok:
        print("\n아직 차단 중입니다. 아무것도 수집하지 않고 종료합니다.")
        print("차단 상태에서 재시도하면 차단이 길어집니다. 몇 시간 뒤 다시 실행하세요.")
        return 2

    print("\n차단이 풀렸습니다.")
    if args.check_only:
        return 0

    print(f"\n과거 유니버스 복원 (2018~2022) · 속도 제한 {krx.MIN_INTERVAL}초/호출")
    dates = hist_dates()
    try:
        df = universe.build_history(dates)
    except Exception as exc:
        print(f"  실패: {exc}")
        return 1

    store.save(df, "universe_history_2018_2022")
    inc = df[df["included"]]
    hist_t = set(inc["ticker"])
    try:
        cur = store.load("universe_20260905")
        cur_t = set(cur[cur["included"]]["ticker"])
    except Exception:
        cur_t = set()

    print(f"\n  스냅샷 {df['asof_date'].nunique()}/{len(dates)}개 성공")
    print(f"  과거 유니버스 고유 종목 {len(hist_t)}개")
    if cur_t:
        print(f"  현재 200종목과 겹침 {len(hist_t & cur_t)}개 · "
              f"과거에만 있던 종목 {len(hist_t - cur_t)}개")
        print(f"  검증에 필요한 합집합 {len(hist_t | cur_t)}개")
    print(f"  KRX 호출 {krx.call_count()}회")

    print("\n다음 단계는 사람이 시작하세요 — 일일 한도를 쓰는 작업입니다.")
    print("  1) DART 재무: 합집합 종목 × 2017~2022  (약 2만 회, 이틀 분량)")
    print("  2) KRX 시세: 합집합 종목 × 2017~2022  (약 400회, 하루 한 번만)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
