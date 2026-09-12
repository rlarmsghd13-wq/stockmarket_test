"""계정 파싱 검증 — KRX 공표값과 대조.

KRX의 PER/EPS/BPS는 **최근 연간(FY) 실적** 기준이다. 우리 스크리너는 TTM을 쓰므로
두 값을 그냥 비교하면 실적이 빠르게 변한 종목에서 크게 어긋난다 — 이건 오류가 아니라
서로 다른 측정이다.

따라서 여기서는 **우리도 FY 기준으로 다시 계산해** 같은 잣대로 비교한다.
이 비교가 맞으면 계정 매칭·금액 파싱이 옳다는 뜻이고,
TTM 로직은 tests/test_pipeline.py 가 따로 검증한다.

    python scripts/99_verify_against_krx.py --n 30
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

from src import dart, krx, quarterly, shares, store  # noqa: E402

TOL = 0.10   # 10% 이내면 통과. 결산 시점·주식수 기준 차이로 소폭 어긋난다.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--universe", default="universe_20260905")
    ap.add_argument("--fy", type=int, default=2025, help="비교 기준 사업연도")
    args = ap.parse_args()

    uni = store.load(args.universe)
    bd = uni["price_date"].iloc[0]
    kf = pd.concat([krx.fundamental_snapshot(bd, m) for m in ("KOSPI", "KOSDAQ")])
    cc = dart.corp_codes()

    targets = (uni[uni["included"]].sort_values("market_cap", ascending=False)
               .head(args.n))
    corp_map = dict(cc[["ticker", "corp_code"]].values)
    sh_all = shares.total_shares(uni)
    sh = shares.with_treasury(
        sh_all[sh_all["ticker"].isin(targets["ticker"])], corp_map, args.fy)

    rows = []
    for u in targets.itertuples():
        code = cc[cc.ticker == u.ticker]
        if code.empty:
            continue
        q = quarterly.fetch_company(code.iloc[0].corp_code, u.ticker, [2024, 2025])
        if q.empty:
            continue
        annual = q[q["quarter"] == 4].sort_values("year")
        if annual.empty:
            continue
        a = annual.iloc[-1]
        # NaN은 파이썬에서 truthy라 `a or b` 로는 폴백이 안 된다.
        pick = lambda *ks: next(                                    # noqa: E731
            (a[k] for k in ks if k in a and pd.notna(a[k])), None)
        ni = pick("net_income_parent", "net_income")
        eq = pick("equity_parent", "equity")
        # 신종자본증권은 회계상 자본이지만 보통주 몫이 아니다 (금융지주에 크다).
        hyb = pick("hybrid_capital") or 0.0
        if eq is not None:
            eq = eq - hyb
        s = sh[sh.ticker == u.ticker]
        if s.empty or pd.isna(ni):
            continue
        s = s.iloc[0]
        # 분모는 유통주식수 = 상장(보통+우선) − 자기주식
        denom = s.shares_outstanding
        eps = ni / denom
        bps = eq / denom if pd.notna(eq) else None
        per = s.market_cap_total / ni if ni > 0 else None
        pbr = s.market_cap_total / eq if pd.notna(eq) and eq > 0 else None

        k = kf[kf.ticker == u.ticker]
        if k.empty:
            continue
        k = k.iloc[0]
        rows.append({
            "ticker": u.ticker, "name": u.name, "fy": int(a.year),
            "eps": eps, "eps_krx": float(k.eps),
            "bps": bps, "bps_krx": float(k.bps),
            "per": per, "per_krx": float(k.per) or None,
            "pbr": pbr, "pbr_krx": float(k.pbr),
            "pref": bool(s.has_preferred),
            "treas": float(s.treasury_ratio),
            "hyb": float(hyb),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("비교 대상이 없습니다.")
        return 1

    def rel(a, b):
        if pd.isna(a) or pd.isna(b) or b in (0, None):
            return None
        return abs(a - b) / abs(b)

    for f in ("eps", "bps", "pbr"):
        df[f"d_{f}"] = [rel(r[f], r[f"{f}_krx"]) for _, r in df.iterrows()]

    print(f"{'종목':<13}{'FY':>5}{'EPS 계산':>11}{'EPS KRX':>10}{'차이':>8}"
          f"{'BPS 계산':>11}{'BPS KRX':>10}{'차이':>8}  우선주 자사주")
    print("-" * 90)
    for r in df.itertuples():
        fmt = lambda v: f"{v:,.0f}" if pd.notna(v) else "—"      # noqa: E731
        pct = lambda v: f"{v*100:5.1f}%" if v is not None and pd.notna(v) else "   —"  # noqa: E731
        print(f"{r.name[:12]:<13}{r.fy:>5}{fmt(r.eps):>11}{fmt(r.eps_krx):>10}{pct(r.d_eps):>8}"
              f"{fmt(r.bps):>11}{fmt(r.bps_krx):>10}{pct(r.d_bps):>8}"
              f"  {'O' if r.pref else ' '}  {r.treas*100:5.1f}%")

    print()
    for f in ("eps", "bps", "pbr"):
        d = df[f"d_{f}"].dropna()
        if len(d):
            ok = (d <= TOL).sum()
            print(f"  {f.upper():<4} {len(d)}종목 중 {ok}종목 오차 {TOL*100:.0f}% 이내"
                  f" (중앙값 {d.median()*100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
