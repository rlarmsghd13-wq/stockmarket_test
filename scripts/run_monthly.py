"""월 1회 전체 실행 — 이것만 돌리면 된다.

    python scripts/run_monthly.py                  # 오늘 기준
    python scripts/run_monthly.py --asof 2026-10-05

지표 → 등급화 → 스코어 → 증권사 목표가 → 볼트 → 성과추적까지 한 번에.
재무·시세는 이미 받아둔 것을 쓰므로 새 분기 공시가 나왔을 때만
03/05 스크립트를 다시 돌리면 된다.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(script: str, *extra: str, soft: bool = False) -> None:
    """soft=True면 실패해도 나머지 단계를 계속한다 (외부 API에 의존하는 단계용)."""
    cmd = [sys.executable, os.path.join(ROOT, "scripts", script), *extra]
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = [l for l in (p.stdout or "").splitlines()
            if l.strip() and "FutureWarning" not in l and "df.replace" not in l
            and "로그인" not in l and "만료" not in l]
    print("\n".join(tail[-14:]))
    if p.returncode != 0:
        print(p.stderr[-800:])
        if soft:
            print(f"  [건너뜀] {script} 실패 — 이 단계 없이 계속합니다")
            return
        raise SystemExit(f"{script} 실패")


def grade(asof: str) -> None:
    from src import grading, store
    m = store.load("metrics_latest")
    tr = store.load("tracks")
    sector = {r.ticker: str(r.induty_code)[:2] for r in tr.itertuples()}
    g = grading.build(m, sector)
    store.save(g, "grades_latest")
    store.save(g, f"grades_{asof.replace('-', '')}")
    print(f"  등급화 {len(g)}종목 · 5개 부문")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    print(f"■ {args.asof} 월간 실행\n")
    print("[1/6] 지표 계산")
    run("04_build_metrics.py", "--asof", args.asof)
    print("\n[2/6] 등급화")
    grade(args.asof)
    print("\n[3/6] 전략 스코어")
    run("06_score_strategies.py", "--asof", args.asof, "--top", str(args.top))
    # 뉴스에 인용된 증권사 목표가. 네이버 키가 없거나 API가 막히면 건너뛰고 계속한다
    # — 참고 정보라 이것 때문에 월간 실행이 멈출 이유가 없다.
    print("\n[4/6] 증권사 목표가 수집")
    run("21_news_consensus.py", soft=True)
    print("\n[5/6] 볼트 갱신")
    run("08_build_vault.py", "--asof", args.asof, "--top", str(args.top))
    # 직전 달 선택을 정산하고 이번 달을 기록한다. 백테스트(17·18)가 전부
    # 인샘플이라, 오염되지 않은 표본은 이렇게 매달 쌓는 수밖에 없다.
    print("\n[6/6] 성과 추적")
    run("19_track.py", "--asof", args.asof, "--top", str(args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
