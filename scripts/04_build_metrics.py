"""L2 — 전 종목 파생 지표 계산.

    python scripts/04_build_metrics.py               # 오늘 시점
    python scripts/04_build_metrics.py --asof 2026-05-20

시점 잠금: 각 종목의 재무는 `rcept_dt <= asof_date` 인 것 중 최신만 쓴다.
그 시점에 실제로 알 수 있었던 값만 들어간다.
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

import config                                                  # noqa: E402
from src import dart, flags, metrics, momentum, shares, store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--universe", default="universe_20260905")
    ap.add_argument("--fy", type=int, default=2025, help="자기주식 조회 기준 사업연도")
    args = ap.parse_args()

    uni = store.load(args.universe)
    inc = uni[uni["included"]]
    tracks = store.load("tracks").set_index("ticker")
    ttm = store.load("fin_ttm")

    cc = dart.corp_codes()
    corp_map = dict(cc[["ticker", "corp_code"]].values)
    sh_all = shares.total_shares(uni)
    sh = shares.with_treasury(
        sh_all[sh_all["ticker"].isin(inc["ticker"])], corp_map, args.fy
    ).set_index("ticker")

    rows, skipped = [], []
    for u in inc.itertuples():
        hist = ttm[ttm["ticker"] == u.ticker]
        if hist.empty:
            skipped.append((u.ticker, u.name, "재무 없음"))
            continue

        # 시점 잠금 — 그날 알 수 있었던 최신 재무만
        visible = hist[pd.to_datetime(hist["rcept_dt"]) <= pd.Timestamp(args.asof)]
        if visible.empty:
            skipped.append((u.ticker, u.name, "해당 시점 공시 없음"))
            continue
        last = visible.sort_values("period").iloc[-1]

        if u.ticker not in sh.index or u.ticker not in tracks.index:
            skipped.append((u.ticker, u.name, "주식수/트랙 없음"))
            continue
        s = sh.loc[u.ticker]
        tr = tracks.loc[u.ticker]

        m = metrics.compute(
            last.to_dict(),
            market_cap=float(s.market_cap_total),
            shares_out=float(s.shares_outstanding),
            hist=visible,
            track=str(tr.track),
        )

        # 안정성 왜곡 종목은 해당 지표를 NULL로. 세그먼트 데이터가 없으면 추정하지 않는다.
        if bool(tr.stability_distorted):
            for k in ("debt_ratio", "current_ratio", "net_debt", "net_debt_ebit",
                      "net_debt_ebitda"):
                m[k] = None

        f = flags.restructure_flags(visible)
        annual = visible[visible["quarter"] == 4]
        streak = flags.loss_streak(annual)

        rows.append({
            "asof_date": args.asof, "ticker": u.ticker, "name": u.name,
            "market": u.market, "track": str(tr.track),
            "market_cap": float(s.market_cap_total),
            "shares_outstanding": float(s.shares_outstanding),
            "treasury_ratio": float(s.treasury_ratio),
            "fin_year": int(last.year), "fin_quarter": int(last.quarter),
            "rcept_dt": last.rcept_dt,
            "loss_streak": streak,
            "loss_flag": flags.loss_flag(
                streak, track=str(tr.track),
                debt_ratio=m.get("debt_ratio"), ocf=last.get("ocf_ttm")),
            "ttm_reliable": flags.ttm_reliable(f, u.ticker, int(last.period)),
            "stability_distorted": bool(tr.stability_distorted),
            **m,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("계산된 지표가 없습니다.")
        return 1

    # 모멘텀·수급 — 저장된 시계열을 기준일로 잘라 계산한다. KRX 호출 없음.
    if store.exists("prices_daily"):
        caps = dict(zip(df["ticker"], df["market_cap"]))
        fl = store.load("flows_daily") if store.exists("flows_daily") else None
        mom = momentum.build(store.load("prices_daily"), fl, caps, args.asof)
        df = df.merge(mom, on="ticker", how="left")
        cov = df["return_3m"].notna().mean() * 100
        print(f"모멘텀 지표 결합: {len(mom)}종목 (3개월 수익률 커버리지 {cov:.0f}%)")
    else:
        print("prices_daily 없음 — 모멘텀 부문은 NULL입니다. "
              "scripts/05_fetch_prices.py 를 먼저 실행하세요.")

    store.save(df, f"metrics_{args.asof.replace('-', '')}")
    store.save(df, "metrics_latest")

    print(f"기준일 {args.asof} · {len(df)}종목 계산" +
          (f" / 제외 {len(skipped)}종목" if skipped else ""))
    for t, n, why in skipped:
        print(f"  [제외] {t} {n}: {why}")

    print("\n트랙별 지표 결측률:")
    key = ["per", "pbr", "roe", "operating_margin", "debt_ratio",
           "fcf_to_ni", "revenue_growth_yoy", "ev_ebit"]
    hdr = "  " + f"{'트랙':<6}{'n':>4}" + "".join(f"{k[:11]:>13}" for k in key)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for tr, g in df.groupby("track"):
        line = f"  {tr:<6}{len(g):>4}"
        for k in key:
            line += f"{g[k].isna().mean()*100:>12.0f}%"
        print(line)

    print(f"\nTTM 신뢰 불가(구조변경): {(~df.ttm_reliable).sum()}종목")
    print(f"적자 이력 보유: {(df.loss_streak > 0).sum()}종목")
    if (df.loss_streak > 0).any():
        for r in df[df.loss_streak > 0].sort_values("loss_streak", ascending=False).itertuples():
            print(f"  {r.ticker} {r.name:<14} {r.loss_streak}년 → {r.loss_flag} ({r.track})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
