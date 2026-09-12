"""게이트 검증 — 199종목을 25종목으로 자르는 필터가 실제로 도움이 되는가.

    python scripts/14_validate_gates.py

게이트는 이 시스템에서 가장 강한 필터인데 한 번도 검증한 적이 없다.
점수(가중치)는 97개월 패널로 검증해 재조정했지만, 게이트는 논리값 그대로다.

방법
    ① 통과군 vs 탈락군의 3개월 순방향 수익률 차이
    ② 게이트를 켠 상위 5종목 vs 끈 상위 5종목 (실제 매수 대상 비교)
    ③ 개별 조건별 기여도 — 그 조건 하나만 껐을 때 무엇이 달라지는가
    ④ 통과 종목 수 시계열 (시장 국면 지표가 되는가)

한계
    월별 코호트가 겹치므로 t값은 과장된다. 날짜 클러스터 보정을 같이 낸다.
"""
from __future__ import annotations

import math
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

TOP_N = 5


# ---------------------------------------------------------------------------
# 이력 기반 게이트 특성을 (종목, 공시) 단위로 한 번에 계산한다.
# strategies.gate_features()를 97번 부르면 같은 계산을 반복하게 된다.
# ---------------------------------------------------------------------------
def rolling_gate_features(ttm: pd.DataFrame) -> pd.DataFrame:
    out = []
    for t, g in ttm.groupby("ticker"):
        g = g.sort_values("period").reset_index(drop=True)
        rev = g["revenue_ttm"] if "revenue_ttm" in g.columns else pd.Series(dtype=float)
        for i in range(len(g)):
            h = g.iloc[: i + 1]
            annual = h[h["quarter"] == 4].tail(3)
            op = annual["operating_income_ttm"] if "operating_income_ttm" in annual else pd.Series(dtype=float)
            fcf = annual["fcf_ttm"] if "fcf_ttm" in annual else pd.Series(dtype=float)

            gq = 0
            for k in range(1, 5):
                j = i - k + 1
                if j - 4 >= 0 and len(rev) > j:
                    cur, prev = rev.iloc[j], rev.iloc[j - 4]
                    if pd.notna(cur) and pd.notna(prev) and prev > 0 and cur > prev:
                        gq += 1

            ann_rev = (annual["revenue_ttm"].dropna() if "revenue_ttm" in annual
                       else pd.Series(dtype=float))
            decline3 = bool(len(ann_rev) >= 3 and (ann_rev.diff().dropna() < 0).all())

            ocf_bad = 0
            if {"ocf_ttm", "net_income_ttm"}.issubset(annual.columns):
                for _, r in annual.tail(2).iterrows():
                    ni, ocf = r.get("net_income_ttm"), r.get("ocf_ttm")
                    if pd.notna(ni) and pd.notna(ocf) and ni > 0 and ocf / ni < 0.5:
                        ocf_bad += 1

            out.append({
                "ticker": t,
                "op_loss_years_3y": int((op.dropna() < 0).sum()),
                "fcf_pos_years_3y": int((fcf.dropna() > 0).sum()),
                "annual_count": int(len(annual)),
                "rev_growth_q_of_4": gq,
                "revenue_decline_3y": decline3,
                "ocf_quality_bad_2y": ocf_bad >= 2,
                "rcept": pd.to_datetime(str(g["rcept_dt"].iloc[i]),
                                        format="%Y%m%d", errors="coerce"),
            })
    d = pd.DataFrame(out)
    return d.dropna(subset=["rcept"]).sort_values("rcept").reset_index(drop=True)


def attach(panel: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """각 기준일에 그 시점까지 공시된 마지막 특성을 붙인다 (시점 잠금)."""
    p = panel.copy()
    p["asof_ts"] = pd.to_datetime(p["asof"])
    return pd.merge_asof(
        p.sort_values("asof_ts"), feats.sort_values("rcept"),
        left_on="asof_ts", right_on="rcept", by="ticker",
        direction="backward")


# ---------------------------------------------------------------------------
# 게이트 조건을 하나씩 끄면서 평가할 수 있도록 분해한다.
# 각 함수는 "탈락이면 True".
# ---------------------------------------------------------------------------
def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


VALUE_GATES = {
    "영업적자 2회+": lambda d: (_num(d["annual_count"]).fillna(0) >= 3)
                              & (_num(d["op_loss_years_3y"]).fillna(0) > 1),
    "부채비율 200%": lambda d: _num(d["debt_ratio"]) > 200,
    "FCF흑자 2회미만": lambda d: (_num(d["annual_count"]).fillna(0) >= 3)
                                & (_num(d["fcf_pos_years_3y"]).fillna(9) < 2),
    "ROE 8% 미만": lambda d: _num(d["roe"]) < 8,
    "시총 5천억 미만": lambda d: _num(d["market_cap"]).fillna(0) < 5_000e8,
    "밸류 상위40% 밖": lambda d: _num(d["valuation_score"]) < 60,
    "퀄리티 상위50% 밖": lambda d: _num(d["quality_score"]) < 50,
}

GROWTH_GATES = {
    "성장성 상위20% 밖": lambda d: _num(d["growth_score"]).isna()
                                  | (_num(d["growth_score"]) < 80),
    "4분기중 성장 3회미만": lambda d: _num(d["rev_growth_q_of_4"]).fillna(0) < 3,
    "유동비율 100% 미만": lambda d: _num(d["current_ratio"]) < 100,
    "시총 3천억 미만": lambda d: _num(d["market_cap"]).fillna(0) < 3_000e8,
}


def base_score(df: pd.DataFrame, strategy: str) -> np.ndarray:
    w = strategies.WEIGHTS[strategy]
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for cat, weight in w.items():
        if not weight:
            continue
        s = _num(df[f"{cat}_score"])
        ok = s.notna()
        num = num + s.fillna(0) * weight * ok
        den = den + weight * ok
    return np.where(den > 0, num / den, np.nan)


def gate_mask(df: pd.DataFrame, gates: dict, skip: str | None = None) -> pd.Series:
    """모든 조건을 통과하면 True. 결측은 통과로 본다 (원 구현과 동일)."""
    ok = pd.Series(True, index=df.index)
    for nm, fn in gates.items():
        if nm == skip:
            continue
        ok &= ~fn(df).fillna(False).astype(bool)
    return ok


def cluster_t(g: pd.DataFrame, col: str = "excess") -> tuple[float, float, int]:
    """월별 평균의 t검정 — 같은 달 종목은 독립이 아니므로 날짜 단위로 묶는다."""
    m = g.groupby("asof")[col].mean().dropna()
    if len(m) < 3 or m.std(ddof=1) == 0:
        return float("nan"), float("nan"), len(m)
    t = m.mean() / (m.std(ddof=1) / math.sqrt(len(m)))
    p = math.erfc(abs(t) / math.sqrt(2))          # 정규근사 양측
    return float(t), float(p), len(m)


def main() -> int:
    panel = store.load("validation_panel_63")
    ttm = pd.concat([store.load("fin_ttm_hist"), store.load("fin_ttm")],
                    ignore_index=True).drop_duplicates(["ticker", "period"],
                                                       keep="last")
    print(f"패널 {panel.shape[0]:,}행 · {panel['ticker'].nunique()}종목 · "
          f"{panel['asof'].nunique()}개월 ({panel['asof'].min()} ~ {panel['asof'].max()})")

    print("게이트 특성 계산 중...")
    feats = rolling_gate_features(ttm)
    df = attach(panel, feats)
    print(f"  특성 {len(feats):,}건 · 결합 후 {len(df):,}행")

    df = df[_num(df["fwd_63"]).notna()].copy()
    df["fwd_63"] = _num(df["fwd_63"])
    df["mkt"] = df.groupby("asof")["fwd_63"].transform("mean")
    df["excess"] = df["fwd_63"] - df["mkt"]

    results = {}
    for strat, gates in ((strategies.VALUE, VALUE_GATES),
                         (strategies.GROWTH, GROWTH_GATES)):
        d = df.copy()
        d["score"] = base_score(d, strat)
        d = d[pd.notna(d["score"])].copy()
        d["pass"] = gate_mask(d, gates)
        results[strat] = (d, gates)

    # ── ① 통과군 vs 탈락군 ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("① 게이트 통과군 vs 탈락군 — 3개월 순방향 수익률")
    print("=" * 78)
    for strat, (d, _) in results.items():
        lab = strategies.LABEL[strat]
        a, b = d[d["pass"]], d[~d["pass"]]
        print(f"\n[{lab}]  통과 {len(a):,}행 ({len(a) / len(d) * 100:.1f}%)"
              f" · 탈락 {len(b):,}행")
        print(f"  {'':<8}{'평균수익률':>13}{'초과수익':>12}{'승률':>9}")
        for nm, x in (("통과", a), ("탈락", b)):
            if x.empty:
                continue
            print(f"  {nm:<8}{x['fwd_63'].mean():>12.2f}%"
                  f"{x['excess'].mean():>11.2f}%"
                  f"{(x['fwd_63'] > 0).mean() * 100:>8.1f}%")
        t, p, n = cluster_t(a)
        print(f"  통과군 초과수익 t={t:.2f} p={p:.3f} (월 표본 {n})")

    # ── ② 실제 매수 대상 비교 ─────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"② 실제 매수 대상 — 매월 점수 상위 {TOP_N}종목")
    print("=" * 78)
    for strat, (d, _) in results.items():
        lab = strategies.LABEL[strat]
        print(f"\n[{lab}]")
        print(f"  {'':<14}{'월수':>6}{'평균수익':>11}{'초과수익':>11}"
              f"{'승률':>8}{'t':>7}{'p':>8}")
        for nm, sub in (("게이트 켬", d[d["pass"]]), ("게이트 끔", d)):
            picks = (sub.sort_values("score", ascending=False)
                        .groupby("asof").head(TOP_N))
            t, p, n = cluster_t(picks)
            print(f"  {nm:<14}{n:>6}{picks['fwd_63'].mean():>10.2f}%"
                  f"{picks['excess'].mean():>10.2f}%"
                  f"{(picks['fwd_63'] > 0).mean() * 100:>7.1f}%"
                  f"{t:>7.2f}{p:>8.3f}")

    # ── ③ 조건별 기여도 ──────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"③ 조건별 기여도 — 그 조건 하나만 껐을 때 상위 {TOP_N}종목 초과수익")
    print("=" * 78)
    for strat, (d, gates) in results.items():
        lab = strategies.LABEL[strat]
        full = (d[d["pass"]].sort_values("score", ascending=False)
                            .groupby("asof").head(TOP_N)["excess"].mean())
        print(f"\n[{lab}]  전체 게이트 적용 시 {full:+.2f}%")
        rows = []
        for nm in gates:
            m = gate_mask(d, gates, skip=nm)
            picks = (d[m].sort_values("score", ascending=False)
                         .groupby("asof").head(TOP_N))
            rows.append((nm, picks["excess"].mean(), m.mean() * 100))
        rows.sort(key=lambda x: x[1])
        print(f"  {'조건을 끄면':<22}{'초과수익':>10}{'변화':>10}{'통과율':>9}")
        for nm, v, rate in rows:
            mark = ("  ← 도움됨" if v < full - 0.05 else
                    "  ← 해로움" if v > full + 0.05 else "")
            print(f"  {nm:<22}{v:>9.2f}%{v - full:>+9.2f}%{rate:>8.1f}%{mark}")

    # ── ④ 통과 종목 수 시계열 ────────────────────────────────────────
    print("\n" + "=" * 78)
    print("④ 게이트 통과 종목 수 — 시장 국면 지표가 되는가")
    print("=" * 78)
    for strat, (d, _) in results.items():
        lab = strategies.LABEL[strat]
        cnt = d.groupby("asof").agg(n=("pass", "sum"), tot=("pass", "size"),
                                    fwd=("mkt", "first"))
        cnt["rate"] = cnt["n"] / cnt["tot"] * 100
        c = cnt[["rate", "fwd"]].dropna()
        corr = (c["rate"].rank().corr(c["fwd"].rank()) if len(c) > 5
                else float("nan"))
        print(f"\n[{lab}]  통과율 중앙값 {cnt['rate'].median():.1f}%"
              f"  (최저 {cnt['rate'].min():.1f}% ~ 최고 {cnt['rate'].max():.1f}%)")
        print(f"  통과율 vs 이후 3개월 시장수익률 순위상관 {corr:+.3f}")
        q = c["rate"].quantile([0.33, 0.67]).tolist()
        lo = c[c["rate"] <= q[0]]["fwd"].mean()
        hi = c[c["rate"] >= q[1]]["fwd"].mean()
        print(f"  통과율 하위 1/3 구간 → 이후 시장 {lo:+.2f}%"
              f"  ·  상위 1/3 구간 → 이후 시장 {hi:+.2f}%")

    out = pd.concat([d.assign(strategy=s) for s, (d, _) in results.items()],
                    ignore_index=True)
    store.save(out, "gate_panel")
    print(f"\n저장: gate_panel ({len(out):,}행)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
