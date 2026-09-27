"""L0 — 유니버스 200종목 트랙 판정.

    python scripts/02_assign_tracks.py            # 편입 200종목
    python scripts/02_assign_tracks.py --n 30     # 상위 30종목만 (빠른 확인)
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import config                                            # noqa: E402
from src import dart, quarterly, store, tracks           # noqa: E402

FY = 2025


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="universe_20260905")
    ap.add_argument("--n", type=int, default=0, help="0이면 편입 전체")
    args = ap.parse_args()

    uni = store.load(args.universe)
    inc = uni[uni["included"]].sort_values("market_cap", ascending=False)
    if args.n:
        inc = inc.head(args.n)
    cc = dart.corp_codes()

    rows, missing = [], []
    for i, u in enumerate(inc.itertuples(), 1):
        c = cc[cc.ticker == u.ticker]
        if c.empty:
            missing.append(u.ticker)
            continue
        code = c.iloc[0].corp_code
        try:
            q = quarterly.fetch_company(code, u.ticker, [FY])
            # 인프라 투융자회사(맥쿼리인프라 등)는 일반 재무제표 구조가 아니라
            # 빈 결과가 온다. 업종코드만으로 판정하고 넘어간다.
            annual = q[q["quarter"] == 4] if "quarter" in q.columns else q
            annual_row = annual.iloc[0].to_dict() if not annual.empty else None
            raw_bs = dart.financials(code, FY, config.REPRT_ANNUAL, "CFS")
            if not raw_bs.empty:
                raw_bs = raw_bs[raw_bs["sj_div"] == "BS"]
            rows.append(tracks.assign(code, u.ticker, u.name,
                                      annual_row=annual_row, raw_bs=raw_bs))
        except Exception as exc:
            print(f"  [실패] {u.ticker} {u.name}: {exc}")
        if i % 25 == 0:
            print(f"  ... {i}/{len(inc)}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("판정 결과가 없습니다.")
        return 1
    df = df.merge(inc[["ticker", "name", "market_cap"]], on="ticker", how="left")
    # **덮어쓰지 않고 합친다.** 과거 유니버스 종목(26_fill_new_tickers.py가 채운
    # 284개)까지 들어 있는 표를 현재 200종목으로 덮으면, 판정 없는 종목이
    # 조용히 T1(일반)으로 취급되어 은행·지주의 지표가 망가진다.
    if store.exists("tracks"):
        old = store.load("tracks")
        df = (pd.concat([df, old], ignore_index=True)
                .drop_duplicates(["ticker"], keep="first"))   # 새 판정이 이긴다
    store.save(df, "tracks")

    print(f"\n판정 {len(df)}종목" + (f" / corp_code 없음 {len(missing)}종목" if missing else ""))
    print("\n트랙 분포:")
    for tr, n in df["track"].value_counts().items():
        cap = df[df.track == tr]["market_cap"].sum() / 1e12
        print(f"  {tr}  {n:>3}종목   시총 합계 {cap:>8,.0f}조")

    fin = df[df["has_finance_segment"]]
    if not fin.empty:
        print(f"\n제조 + 금융 복합 {len(fin)}종목:")
        for r in fin.sort_values("finance_asset_ratio", ascending=False).itertuples():
            mark = ("안정성 왜곡 → 세그먼트 필요" if r.stability_distorted
                    else "부채비율 정상 — 지표 유지")
            print(f"  {r.ticker} {r.name:<14} 금융자산 {r.finance_asset_ratio*100:>5.1f}%"
                  f"  부채비율 {r.debt_ratio:>6.0f}%   {mark}")

    for tr in (config.TRACK_FINANCE, config.TRACK_PHARMA, config.TRACK_HOLDING):
        sub = df[df.track == tr].sort_values("market_cap", ascending=False).head(8)
        if sub.empty:
            continue
        print(f"\n{tr} 상위:")
        for r in sub.itertuples():
            print(f"  {r.name:<16} {r.induty_code:<6} {r.track_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
