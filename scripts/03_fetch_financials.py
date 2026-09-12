"""L1 — 200종목 재무 적재 + TTM 집계.

    python scripts/03_fetch_financials.py

DART 원본은 data/raw/dart/ 에 JSON으로 남으므로 중단 후 다시 실행해도
이미 받은 종목은 네트워크를 타지 않는다. 안심하고 끊었다 이어도 된다.

산출: fin_quarterly.parquet, fin_ttm.parquet, fetch_report.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import config                                                    # noqa: E402
from src import dart, flags, quarterly, store, ttm               # noqa: E402


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="universe_20260905")
    ap.add_argument("--n", type=int, default=0, help="0이면 편입 전체")
    args = ap.parse_args()

    this_year = datetime.now().year
    years = list(range(this_year - config.FIN_YEARS_BACK + 1, this_year + 1))
    log(f"수집 연도 {years[0]}~{years[-1]} ({len(years)}년)")

    uni = store.load(args.universe)
    inc = uni[uni["included"]].sort_values("market_cap", ascending=False)
    if args.n:
        inc = inc.head(args.n)
    cc = dart.corp_codes()
    corp_map = dict(cc[["ticker", "corp_code"]].values)

    q_all, t_all, report = [], [], []
    t0 = time.monotonic()

    for i, u in enumerate(inc.itertuples(), 1):
        code = corp_map.get(u.ticker)
        rec = {"ticker": u.ticker, "name": u.name, "rows": 0,
               "ttm_rows": 0, "loss_streak": 0, "restructure": 0,
               "status": "ok", "note": ""}
        if not code:
            rec.update(status="skip", note="corp_code 없음")
            report.append(rec)
            continue
        try:
            q = quarterly.fetch_company(code, u.ticker, years)
            if q.empty or "quarter" not in q.columns:
                rec.update(status="empty", note="재무 없음 (펀드형 등)")
                report.append(rec)
                continue
            t = ttm.build(q)
            f = flags.restructure_flags(t)
            annual = t[t["quarter"] == 4]

            q_all.append(q)
            t_all.append(t)
            rec.update(
                rows=len(q), ttm_rows=len(t),
                loss_streak=flags.loss_streak(annual),
                # 전파된 "직후" 행은 빼고 실제 사건만 센다
                restructure=int((~f["note"].str.contains("직후")).sum()) if len(f) else 0,
            )
        except Exception as exc:
            rec.update(status="fail", note=f"{type(exc).__name__}: {exc}")
            log(f"  [실패] {u.ticker} {u.name}: {exc}")
            traceback.print_exc(limit=1)
        report.append(rec)

        if i % 10 == 0:
            el = time.monotonic() - t0
            eta = el / i * (len(inc) - i)
            log(f"  {i}/{len(inc)}  경과 {el/60:.1f}분  남은 예상 {eta/60:.1f}분")

    if not q_all:
        log("적재된 재무가 없습니다.")
        return 1

    fq = pd.concat(q_all, ignore_index=True)
    ft = pd.concat(t_all, ignore_index=True)
    rep = pd.DataFrame(report)
    store.save(fq, "fin_quarterly")
    store.save(ft, "fin_ttm")
    store.save(rep, "fetch_report")

    log("")
    log(f"완료 — 소요 {(time.monotonic()-t0)/60:.1f}분")
    log(f"  fin_quarterly {len(fq):,}행 / fin_ttm {len(ft):,}행 / 종목 {fq.ticker.nunique()}개")
    log("")
    log("상태 분포:")
    for s, n in rep["status"].value_counts().items():
        log(f"  {s:<6} {n:>3}종목")
    bad = rep[rep.status != "ok"]
    if not bad.empty:
        log("")
        log("정상 아님:")
        for r in bad.itertuples():
            log(f"  {r.ticker} {r.name:<14} {r.status:<6} {r.note}")

    log("")
    log(f"연속 적자 이력: {(rep.loss_streak > 0).sum()}종목")
    for r in rep[rep.loss_streak > 0].sort_values("loss_streak", ascending=False).itertuples():
        fl = flags.loss_flag(r.loss_streak, track="T1")
        log(f"  {r.ticker} {r.name:<14} {r.loss_streak}년 → {fl}")

    log("")
    log(f"구조변경(인적분할·합병 등) 탐지: {(rep.restructure > 0).sum()}종목")
    for r in rep[rep.restructure > 0].sort_values("restructure", ascending=False).head(15).itertuples():
        log(f"  {r.ticker} {r.name:<14} {r.restructure}건")

    log("")
    log("주요 지표 결측률 (fin_ttm 기준):")
    for c in ("revenue_ttm", "operating_income_ttm", "net_income_parent_ttm",
              "ocf_ttm", "fcf_ttm", "equity_parent", "da_ttm"):
        if c in ft.columns:
            log(f"  {c:<26} {ft[c].isna().mean()*100:>5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
