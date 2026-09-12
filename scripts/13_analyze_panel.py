"""저장된 검증 패널 분석 — 코호트 재계산 없이 결과만 다시 본다.

    python scripts/13_analyze_panel.py --horizon 63

09_validate.py가 만든 validation_panel_{horizon} 을 읽어
IC·5분위·전략 종합점수를 계산한다. 분석 기준을 바꿔가며 여러 번 볼 때
30분짜리 코호트 생성을 반복하지 않기 위해 분리했다.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src import store, strategies  # noqa: E402

CATS = ["growth", "profitability", "stability", "valuation", "momentum"]
CAT_KR = {"growth": "성장성", "profitability": "수익성", "stability": "안정성",
          "valuation": "밸류에이션", "momentum": "모멘텀·수급"}
RAW = [("per", "PER", -1), ("pbr", "PBR", -1), ("roe", "ROE", +1),
       ("operating_margin", "영업이익률", +1), ("debt_ratio", "부채비율", -1),
       ("fcf_to_ni", "FCF/순이익", +1), ("ocf_to_ni", "영업CF/순이익", +1),
       ("revenue_growth_yoy", "매출성장률", +1), ("ev_ebit", "EV/EBIT", -1),
       ("revenue_cagr_3y", "매출 3년 CAGR", +1), ("fcf_yield", "FCF수익률", +1),
       ("return_3m", "3개월 수익률", +1), ("volume_ratio", "거래량비", +1),
       ("disparity_20", "20일 이격도", -1)]


def ic(x: pd.Series, y: pd.Series, min_n: int = 20) -> float:
    """스피어만 순위상관. 순위로 바꿔 피어슨을 적용한다 (scipy 불필요)."""
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < min_n:
        return np.nan
    r = d.rank()
    v = float(r.iloc[:, 0].corr(r.iloc[:, 1]))
    return v if pd.notna(v) else np.nan


def verdict(m: float) -> str:
    a = abs(m)
    if a < 0.02:
        return "무의미"
    if m < 0:
        return "역방향" if a >= 0.03 else "약한 역방향"
    return "강함" if a >= 0.10 else ("쓸만함" if a >= 0.05 else "약함")


def by_month(P: pd.DataFrame, col: pd.Series, fwd: str) -> pd.Series:
    tmp = P[[fwd, "asof"]].copy()
    tmp["_x"] = col
    return tmp.groupby("asof").apply(
        lambda d: ic(d["_x"], d[fwd]), include_groups=False).dropna()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=63)
    ap.add_argument("--split", action="store_true", help="전반기/후반기로 나눠서도 본다")
    args = ap.parse_args()

    name = f"validation_panel_{args.horizon}"
    P = store.load(name)
    fwd = f"fwd_{args.horizon}"
    P = P[P[fwd].notna()].copy()
    months = P["asof"].nunique()
    print(f"패널 {len(P):,}행 · {months}개월 · {P['ticker'].nunique()}종목 "
          f"· 평균 {len(P)/months:.0f}종목/월")
    print(f"기간 {P['asof'].min()} ~ {P['asof'].max()} · 순방향 {args.horizon}거래일\n")

    def block(title, items, getcol):
        print("=" * 78)
        print(title)
        print("=" * 78)
        print(f"  {'항목':<18}{'평균 IC':>9}{'중앙값':>9}{'양수 비율':>10}{'표본':>7}   해석")
        print("  " + "-" * 66)
        out = []
        for key, label in items:
            c = getcol(key)
            if c is None:
                continue
            ics = by_month(P, c, fwd)
            if ics.empty:
                continue
            out.append((label, ics.mean(), ics.median(), (ics > 0).mean() * 100, len(ics)))
        for label, m, md, pos, n in sorted(out, key=lambda x: -x[1]):
            print(f"  {label:<18}{m:>+9.3f}{md:>+9.3f}{pos:>9.0f}%{n:>7}   {verdict(m)}")
        print()
        return out

    block("① 부문 점수 IC — 점수 순위 vs 이후 수익률 순위",
          [(c, CAT_KR[c]) for c in CATS],
          lambda c: P[f"{c}_score"] if f"{c}_score" in P.columns else None)

    block("② 개별 지표 IC — 방향 반영 (낮을수록 좋은 지표는 부호 반전)",
          [(c, kr) for c, kr, _ in RAW],
          lambda c: (P[c] * dict((x[0], x[2]) for x in RAW)[c]
                     if c in P.columns else None))

    # ③ 5분위
    print("=" * 78)
    print(f"③ 5분위 평균 수익률 — 각 월 안에서 부문 점수로 5등분 (%, {args.horizon}거래일)")
    print("=" * 78)
    print(f"  {'부문':<14}{'Q1(하위)':>10}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'Q5(상위)':>10}{'Q5−Q1':>9}")
    print("  " + "-" * 62)
    for c in CATS:
        col = f"{c}_score"
        if col not in P.columns:
            continue
        d = P[[col, fwd, "asof"]].dropna()
        if len(d) < 500:
            continue
        d = d.assign(q=d.groupby("asof")[col].transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])))
        mean = d.groupby("q", observed=True)[fwd].mean()
        if len(mean) < 5:
            continue
        cells = "".join(f"{mean.iloc[i]:>9.2f}" if i in (0, 4) else f"{mean.iloc[i]:>8.2f}"
                        for i in range(5))
        print(f"  {CAT_KR[c]:<14}{cells}{mean.iloc[-1]-mean.iloc[0]:>+9.2f}")
    print()

    # ④ 전략 종합점수
    print("=" * 78)
    print("④ 전략 종합점수 IC — 게이트 없이 가중합만으로")
    print("=" * 78)
    for s, w in strategies.WEIGHTS.items():
        num = sum(P[f"{c}_score"].fillna(0) * v for c, v in w.items() if v)
        den = sum(P[f"{c}_score"].notna().astype(float) * v for c, v in w.items() if v)
        col = pd.Series(np.where(den > 0, num / den, np.nan), index=P.index)
        ics = by_month(P, col, fwd)
        if ics.empty:
            continue
        print(f"  {strategies.LABEL[s]:<10} 평균 IC {ics.mean():>+.3f}  "
              f"중앙값 {ics.median():>+.3f}  양수 {(ics>0).mean()*100:>3.0f}%  "
              f"표본 {len(ics)}개월   {verdict(ics.mean())}")
    print()

    if args.split:
        mid = sorted(P["asof"].unique())[months // 2]
        print("=" * 78)
        print(f"⑤ 기간 분할 — 전반기(~{mid}) vs 후반기")
        print("=" * 78)
        print(f"  {'부문':<14}{'전반기 IC':>11}{'후반기 IC':>11}   안정성")
        print("  " + "-" * 48)
        for c in CATS:
            col = f"{c}_score"
            if col not in P.columns:
                continue
            a = by_month(P[P["asof"] < mid], P.loc[P["asof"] < mid, col], fwd)
            b = by_month(P[P["asof"] >= mid], P.loc[P["asof"] >= mid, col], fwd)
            if a.empty or b.empty:
                continue
            same = "일관" if a.mean() * b.mean() > 0 else "부호 뒤집힘"
            print(f"  {CAT_KR[c]:<14}{a.mean():>+11.3f}{b.mean():>+11.3f}   {same}")
        print()

    print("=" * 78)
    print("IC 해석: |IC| < 0.02 무의미 · 0.02~0.05 약함 · 0.05~0.10 쓸만함 · 0.10+ 강함")
    print("월별 코호트가 겹치므로 t값은 과장된다. 평균 IC와 양수 비율만 본다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
