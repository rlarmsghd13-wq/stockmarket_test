"""지표 유효성 검증 — 점수가 실제로 이후 수익률을 예측했는가.

    python scripts/09_validate.py --start 2022-10 --horizon 3

방법
    매월 기준일마다 그 시점에 알 수 있었던 재무로 부문 점수를 내고,
    이후 1·3·6개월 수익률과 짝지어 패널을 만든다. 그리고
      ① IC — 각 월별로 점수 순위와 수익률 순위의 상관(스피어만), 전 기간 평균
      ② 5분위 — 점수 상위 20%와 하위 20%의 평균 수익률 차이
    를 본다. 투신/사모 백테스트에서 "비중이 높을수록 좋다"는 전제가 뒤집혔던 것과
    같은 방식으로, 가중치에 넣은 통념이 실제로 성립하는지 확인한다.

생존편향
    `fin_ttm_hist` / `prices_daily_hist` 가 있으면 **과거 시점의 유니버스**
    (그때 상위였다가 지금은 밀려난 종목 포함)까지 합쳐 계산한다. 없으면
    현재 200종목만 쓰게 되어 "살아남은 회사만" 보는 셈이 되므로,
    그 경우의 결과는 낙관적으로 치우친다.

기타 한계
    월별 코호트가 겹치므로(overlapping) 통계적 유의성은 과장된다.
    t값 대신 평균 IC와 양수 비율만 본다.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import config                                                       # noqa: E402
from src import panel, store, strategies, universe   # noqa: E402

CATS = ["growth", "profitability", "stability", "valuation", "momentum"]
CAT_KR = {"growth": "성장성", "profitability": "수익성", "stability": "안정성",
          "valuation": "밸류에이션", "momentum": "모멘텀·수급"}
# 부문 점수와 별도로, 개별 지표가 실제로 예측력이 있는지도 본다.
RAW = [("per", "PER", -1), ("pbr", "PBR", -1), ("roe", "ROE", +1),
       ("operating_margin", "영업이익률", +1), ("debt_ratio", "부채비율", -1),
       ("fcf_to_ni", "FCF/순이익", +1), ("ocf_to_ni", "영업CF/순이익", +1),
       ("revenue_growth_yoy", "매출성장률", +1), ("ev_ebit", "EV/EBIT", -1),
       ("return_3m", "3개월 수익률", +1), ("volume_ratio", "거래량비", +1)]


def fwd_returns(px: pd.DataFrame, asof: str, horizons=(21, 63, 126)) -> pd.DataFrame:
    """기준일 이후 순방향 수익률. 기준일 종가 대비."""
    rows = []
    ts = pd.Timestamp(asof)
    for t, g in px.groupby("ticker"):
        g = g.sort_values("date")
        past = g[g["date"] <= ts]
        future = g[g["date"] > ts]
        if past.empty:
            continue
        p0 = float(past["close_adj"].iloc[-1])
        if p0 <= 0:
            continue
        rec = {"ticker": t}
        for h in horizons:
            rec[f"fwd_{h}"] = (float(future["close_adj"].iloc[h - 1]) / p0 - 1) * 100 \
                if len(future) >= h else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def ic(x: pd.Series, y: pd.Series) -> float:
    """스피어만 순위상관.

    pandas의 method="spearman"은 scipy를 요구한다. 스피어만은 **순위에 피어슨을
    적용한 것**이므로 직접 순위로 바꿔 계산하면 의존성 없이 같은 값이 나온다.
    """
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 20:
        return np.nan
    r = d.rank()
    v = float(r.iloc[:, 0].corr(r.iloc[:, 1]))
    return v if pd.notna(v) else np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-10")
    ap.add_argument("--horizon", type=int, default=63, choices=[21, 63, 126])
    args = ap.parse_args()

    # 과거 데이터(2017~2022)가 있으면 현재분과 합친다. 없으면 현재분만.
    ttm = store.load("fin_ttm")
    px = store.load("prices_daily")
    if store.exists("fin_ttm_hist"):
        ttm = pd.concat([store.load("fin_ttm_hist"), ttm], ignore_index=True)
        ttm = ttm.drop_duplicates(subset=["ticker", "period"], keep="last")
        print(f"과거 재무 결합 → {len(ttm):,}행 / {ttm.ticker.nunique()}종목")
    if store.exists("prices_daily_hist"):
        px = pd.concat([store.load("prices_daily_hist"), px], ignore_index=True)
        px = px.drop_duplicates(subset=["ticker", "date"], keep="last")
        print(f"과거 시세 결합 → {len(px):,}행 / {px.ticker.nunique()}종목")
    px["date"] = pd.to_datetime(px["date"])
    fl = store.load("flows_daily") if store.exists("flows_daily") else None
    if fl is not None:
        fl["date"] = pd.to_datetime(fl["date"])
    tracks = store.load("tracks").set_index("ticker")
    # 주식수는 과거 + 현재 유니버스 스냅샷을 합쳐 시점별로 찾는다.
    # 현재 유니버스만 쓰면 지금 상위 200에 없는 종목이 통째로 빠진다.
    snapshots = universe.load_snapshots()[["ticker", "asof_date", "shares_out"]]
    print(f"주식수 스냅샷 {len(snapshots):,}행 / {snapshots.ticker.nunique()}종목")
    sector = {t: str(tracks.loc[t, "induty_code"])[:2] for t in tracks.index}

    # 순방향 수익률을 확보할 수 있는 마지막 기준일
    last_px = px["date"].max()
    end = (last_px - timedelta(days=int(args.horizon * 1.5))).strftime("%Y-%m")
    first_px = px["date"].min()
    start = max(args.start, (first_px + timedelta(days=400)).strftime("%Y-%m"))
    dates = panel.month_starts(start, end)
    print(f"시세 {first_px:%Y-%m-%d} ~ {last_px:%Y-%m-%d}")
    print(f"코호트 {len(dates)}개월: {dates[0]} ~ {dates[-1]} · 순방향 {args.horizon}거래일\n")

    cohorts = []
    for i, d in enumerate(dates, 1):
        c = panel.cohort(d, ttm, px, panel.shares_asof(snapshots, d),
                         tracks, sector)
        if c is None:
            continue
        r = fwd_returns(px, d, (args.horizon,))
        c = c.merge(r, on="ticker", how="left")
        cohorts.append(c)
        if i % 6 == 0:
            print(f"  {i}/{len(dates)} 코호트 처리")
    if not cohorts:
        print("패널을 만들지 못했습니다.")
        return 1
    P = pd.concat(cohorts, ignore_index=True)
    store.save(P, f"validation_panel_{args.horizon}")
    fwd = f"fwd_{args.horizon}"
    P = P[P[fwd].notna()]
    print(f"\n패널 {len(P):,}행 · {P['asof'].nunique()}개월 · 평균 {len(P)/P['asof'].nunique():.0f}종목/월")

    # ── ① 부문 점수의 IC ────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print(f"① 부문 점수 IC — 점수 순위 vs {args.horizon}거래일 수익률 순위 (스피어만)")
    print("=" * 74)
    print(f"  {'부문':<14}{'평균 IC':>9}{'중앙값':>9}{'양수 비율':>10}{'표본 월':>8}")
    print("  " + "-" * 52)
    for c in CATS:
        col = f"{c}_score"
        if col not in P.columns:
            continue
        ics = P.groupby("asof").apply(lambda d, cc=col: ic(d[cc], d[fwd]))
        ics = ics.dropna()
        if ics.empty:
            continue
        print(f"  {CAT_KR[c]:<14}{ics.mean():>+9.3f}{ics.median():>+9.3f}"
              f"{(ics > 0).mean()*100:>9.0f}%{len(ics):>8}")

    # ── ② 개별 지표의 IC ────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print("② 개별 지표 IC — 방향 반영 후 (낮을수록 좋은 지표는 부호 반전)")
    print("=" * 74)
    print(f"  {'지표':<16}{'평균 IC':>9}{'양수 비율':>10}{'표본 월':>8}   해석")
    print("  " + "-" * 62)
    res = []
    for col, kr, sign in RAW:
        if col not in P.columns:
            continue
        ics = P.groupby("asof").apply(
            lambda d, cc=col, s=sign: ic(d[cc] * s, d[fwd])).dropna()
        if ics.empty:
            continue
        res.append((kr, ics.mean(), (ics > 0).mean() * 100, len(ics)))
    for kr, m, pos, n in sorted(res, key=lambda x: -x[1]):
        verdict = ("예측력 있음" if m > 0.03 else
                   "역방향" if m < -0.03 else "무의미")
        print(f"  {kr:<16}{m:>+9.3f}{pos:>9.0f}%{n:>8}   {verdict}")

    # ── ③ 5분위 스프레드 ────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print(f"③ 5분위 평균 수익률 — 각 월 안에서 부문 점수로 5등분 (%, {args.horizon}거래일)")
    print("=" * 74)
    print(f"  {'부문':<14}{'Q1(하위)':>10}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'Q5(상위)':>10}{'Q5−Q1':>9}")
    print("  " + "-" * 60)
    for c in CATS:
        col = f"{c}_score"
        if col not in P.columns:
            continue
        d = P[[col, fwd, "asof"]].dropna()
        if d.empty:
            continue
        d = d.assign(q=d.groupby("asof")[col].transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])))
        mean = d.groupby("q", observed=True)[fwd].mean()
        if len(mean) < 5:
            continue
        spread = mean.iloc[-1] - mean.iloc[0]
        print(f"  {CAT_KR[c]:<14}" + "".join(f"{mean.iloc[i]:>9.2f}" if i in (0, 4)
                                             else f"{mean.iloc[i]:>8.2f}" for i in range(5))
              + f"{spread:>+9.2f}")

    # ── ④ 전략 종합점수 ────────────────────────────────────────────────
    print(f"\n{'='*74}")
    print("④ 전략 종합점수 IC — 게이트 없이 가중합만으로")
    print("=" * 74)
    for s, w in strategies.WEIGHTS.items():
        num = sum(P[f"{c}_score"].fillna(0) * v for c, v in w.items() if v)
        den = sum((P[f"{c}_score"].notna()).astype(float) * v for c, v in w.items() if v)
        P[f"score_{s}"] = np.where(den > 0, num / den, np.nan)
        ics = P.groupby("asof").apply(
            lambda d, cc=f"score_{s}": ic(d[cc], d[fwd])).dropna()
        if ics.empty:
            continue
        print(f"  {strategies.LABEL[s]:<10} 평균 IC {ics.mean():>+.3f}"
              f"  양수 비율 {(ics > 0).mean()*100:>3.0f}%  표본 {len(ics)}개월")

    print(f"\n{'='*74}")
    print("IC 해석: |IC| < 0.03 무의미 · 0.03~0.05 약함 · 0.05~0.10 쓸만함 · 0.10+ 강함")
    print("생존편향: 유니버스가 현재 시총 상위 200으로 고정되어 과거 탈락 종목이 빠져 있음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
