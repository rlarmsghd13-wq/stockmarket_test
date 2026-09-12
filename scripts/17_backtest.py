"""계좌 단위 백테스트 — 실제로 매달 사고팔았으면 돈이 어떻게 됐나.

    python scripts/17_backtest.py
    python scripts/17_backtest.py --top 5 --cost 0.35

기존 09_validate는 IC(순위 정확도)만 봤다. "점수가 순위를 맞게 매기나"와
"실제로 얼마를 벌었나"는 다른 질문이고, 매달 용돈으로 사는 상황에서는 후자다.

표 구성은 외부 백테스트 검토(_참고/타인백테스트_검토.md)에서 가져왔다.
    연도별 분해 · t/p · MDD · 승률 · 손익비 · 거래비용 세전세후 · 벤치마크
    그리고 그 글의 결정타였던 **소수 종목 의존도**(6건 빼면 마이너스)

거래비용
    수수료 0.015% ×2 + 증권거래세 0.18%(매도) + 슬리피지 0.15% ≈ 편도 0.35%
"""
from __future__ import annotations

import argparse
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

from src import btdata, store, strategies  # noqa: E402


def mdd(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min() * 100)


def tstat(x: pd.Series) -> tuple[float, float]:
    x = x.dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan"), float("nan")
    t = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
    return float(t), float(math.erfc(abs(t) / math.sqrt(2)))


def simulate(d: pd.DataFrame, rets: pd.DataFrame, top: int, cost: float,
             use_gate: bool, asofs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """매달 상위 top종목 동일가중. 반환: (월별 성과, 개별 보유 기록).

    통과 종목이 0개인 달은 **현금 보유**로 본다. 게이트를 켠 성장주는
    97개월 중 20개월이 그렇다 — 그 달을 표본에서 빼면 시장에서 빠져 있던
    시간이 공짜가 되어 성과가 부풀려진다.
    """
    if use_gate:
        d = d[d["pass"]]
    r = rets.set_index(["asof", "ticker"])["ret_m"]

    held: set[str] = set()
    months, legs = [], []
    for a in asofs:
        cand = (d[d["asof"] == a].sort_values("score", ascending=False)
                                 .head(top))
        picks = list(cand["ticker"])
        if not picks:
            months.append({"asof": a, "ret": 0.0, "n": 0, "turnover": 0.0,
                           "gross": 0.0})
            held = set()
            continue

        raw = []
        for t in picks:
            v = r.get((a, t))
            if v is not None and pd.notna(v):
                raw.append((t, float(v)))
        if not raw:
            months.append({"asof": a, "ret": 0.0, "n": 0, "turnover": 0.0,
                           "gross": 0.0})
            continue

        gross = float(np.mean([v for _, v in raw]))
        new = set(t for t, _ in raw)
        turn = len(new - held) / len(new)          # 신규 편입 비중
        # 편입·이탈 양쪽에 편도 비용
        fee = cost * (len(new - held) + len(held - new)) / max(len(new), 1)
        months.append({"asof": a, "ret": gross - fee, "gross": gross,
                       "n": len(raw), "turnover": turn})
        nm = dict(zip(cand["ticker"], cand["name"]))
        for t, v in raw:
            legs.append({"asof": a, "ticker": t, "name": nm.get(t, t),
                         "ret": v, "new": t not in held})
        held = new

    m = pd.DataFrame(months)
    m["year"] = m["asof"].str[:4]
    m["equity"] = (1 + m["ret"] / 100).cumprod()
    return m, pd.DataFrame(legs)


def bench_monthly(gp: pd.DataFrame, rets: pd.DataFrame,
                  asofs: list[str]) -> pd.DataFrame:
    u = gp[gp["in_universe"]][["asof", "ticker", "market_cap"]].drop_duplicates()
    j = u.merge(rets, on=["asof", "ticker"], how="inner")
    eq = j.groupby("asof")["ret_m"].mean().rename("eq")
    cw = (j.groupby("asof")
           .apply(lambda g: np.average(g["ret_m"], weights=g["market_cap"]),
                  include_groups=False).rename("cw"))
    return pd.concat([eq, cw], axis=1).reindex(asofs).fillna(0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--cost", type=float, default=0.35, help="편도 거래비용 %%")
    args = ap.parse_args()

    gp = btdata.prepare()
    asofs = sorted(gp["asof"].unique())
    print(f"기준일 {len(asofs)}개 ({asofs[0]} ~ {asofs[-1]})")
    print("월간 수익률 계산 중...")
    rets = btdata.monthly_returns(asofs)
    print(f"  {len(rets):,}건 · {rets['ticker'].nunique()}종목")

    bm = bench_monthly(gp, rets, asofs)
    bm["eq_eq"] = (1 + bm["eq"] / 100).cumprod()
    bm["cw_eq"] = (1 + bm["cw"] / 100).cumprod()
    yrs = len(asofs) / 12

    print("\n" + "=" * 92)
    print(f"벤치마크 ({yrs:.1f}년)")
    print("=" * 92)
    for nm, col, eqc in (("유니버스 동일가중", "eq", "eq_eq"),
                         ("유니버스 시총가중(지수근사)", "cw", "cw_eq")):
        tot = (bm[eqc].iloc[-1] - 1) * 100
        print(f"  {nm:<26}총 {tot:>+9.1f}%  CAGR {((1 + tot / 100) ** (1 / yrs) - 1) * 100:>+6.2f}%"
              f"  MDD {mdd(bm[eqc]):>7.2f}%  월승률 {(bm[col] > 0).mean() * 100:>5.1f}%")

    runs = {}
    for strat in (strategies.VALUE, strategies.GROWTH):
        d = gp[(gp["strategy"] == strat) & gp["in_universe"]].copy()
        for gate in (True, False):
            m, legs = simulate(d, rets, args.top, args.cost, gate, asofs)
            runs[(strat, gate)] = (m, legs)

    # ── 전체 성과 ────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print(f"전체 성과 — 매월 상위 {args.top}종목 동일가중 · 편도비용 {args.cost}%")
    print("=" * 92)
    print(f"  {'전략':<22}{'총수익':>10}{'CAGR':>8}{'MDD':>8}{'월승률':>8}"
          f"{'초과(동일)':>11}{'t':>6}{'p':>7}")
    for (strat, gate), (m, _) in runs.items():
        nm = f"{strategies.LABEL[strat]} 게이트{'O' if gate else 'X'}"
        tot = (m["equity"].iloc[-1] - 1) * 100
        exc = m["ret"] - bm["eq"].values
        t, p = tstat(exc)
        print(f"  {nm:<22}{tot:>+9.1f}%{((1 + tot / 100) ** (1 / yrs) - 1) * 100:>+7.2f}%"
              f"{mdd(m['equity']):>7.2f}%{(m['ret'] > 0).mean() * 100:>7.1f}%"
              f"{exc.mean():>10.2f}%{t:>6.2f}{p:>7.3f}")

    # ── 연도별 ───────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("연도별 분해 — 전략 수익률 / 유니버스 대비 초과")
    print("=" * 92)
    for (strat, gate), (m, _) in runs.items():
        if not gate and strat == strategies.VALUE:
            pass
        nm = f"{strategies.LABEL[strat]} 게이트{'O' if gate else 'X'}"
        print(f"\n[{nm}]")
        print(f"  {'연도':<7}{'월수':>5}{'수익률':>10}{'벤치마크':>10}{'초과':>9}"
              f"{'MDD':>8}{'월승률':>8}{'t':>6}{'p':>7}")
        bmy = bm.copy()
        bmy["year"] = [a[:4] for a in bm.index]
        for y, g in m.groupby("year"):
            b = bmy[bmy["year"] == y]
            tot = ((1 + g["ret"] / 100).prod() - 1) * 100
            bt = ((1 + b["eq"] / 100).prod() - 1) * 100
            exc = g["ret"].values - b["eq"].values
            t, p = tstat(pd.Series(exc))
            print(f"  {y:<7}{len(g):>5}{tot:>+9.2f}%{bt:>+9.2f}%{tot - bt:>+8.2f}%"
                  f"{mdd((1 + g['ret'] / 100).cumprod()):>7.2f}%"
                  f"{(g['ret'] > 0).mean() * 100:>7.1f}%{t:>6.2f}{p:>7.3f}")

    # ── 거래 통계 ────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("거래 통계 — 승률·손익비·비용")
    print("=" * 92)
    print(f"  {'전략':<22}{'보유건수':>9}{'승률':>8}{'평균수익':>10}{'평균손실':>10}"
          f"{'손익비':>8}{'손익분기승률':>13}")
    for (strat, gate), (m, legs) in runs.items():
        if legs.empty:
            continue
        nm = f"{strategies.LABEL[strat]} 게이트{'O' if gate else 'X'}"
        w = legs[legs["ret"] > 0]["ret"].mean()
        l = legs[legs["ret"] <= 0]["ret"].mean()
        wr = (legs["ret"] > 0).mean() * 100
        be = -l / (w - l) * 100 if pd.notna(w) and pd.notna(l) else float("nan")
        print(f"  {nm:<22}{len(legs):>9,}{wr:>7.1f}%{w:>9.2f}%{l:>9.2f}%"
              f"{abs(w / l):>8.2f}{be:>12.1f}%")

    print(f"\n  거래비용 영향 (편도 {args.cost}%)")
    for (strat, gate), (m, _) in runs.items():
        nm = f"{strategies.LABEL[strat]} 게이트{'O' if gate else 'X'}"
        gtot = ((1 + m["gross"] / 100).prod() - 1) * 100
        ntot = (m["equity"].iloc[-1] - 1) * 100
        print(f"    {nm:<22}세전 {gtot:>+9.1f}%  →  세후 {ntot:>+9.1f}%"
              f"   (월 회전율 {m['turnover'].mean() * 100:.0f}%)")

    # ── 종목 집중도 ──────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("종목 집중도 — 상위 기여 종목을 빼면 무엇이 남는가")
    print("=" * 92)
    for (strat, gate), (m, legs) in runs.items():
        if legs.empty:
            continue
        nm = f"{strategies.LABEL[strat]} 게이트{'O' if gate else 'X'}"
        contrib = legs.groupby("name")["ret"].sum().sort_values(ascending=False)
        base = legs["ret"].mean()
        print(f"\n[{nm}]  보유 {len(legs)}건 · {legs['name'].nunique()}종목"
              f" · 건당 평균 {base:+.2f}%")
        print("  최다 등장: " + " · ".join(
            f"{k}({v})" for k, v in legs["name"].value_counts().head(5).items()))
        for k in (1, 3, 5):
            drop = set(contrib.head(k).index)
            rest = legs[~legs["name"].isin(drop)]["ret"].mean()
            print(f"  상위 {k}종목 제외 → 건당 평균 {rest:+.2f}%"
                  f"  ({rest - base:+.2f}%p)   [{', '.join(list(drop)[:k])}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
