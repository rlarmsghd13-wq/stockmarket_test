"""단일 지표 vs 복합 점수 — 복잡한 점수가 값을 하는가.

    python scripts/18_factor_race.py
    python scripts/18_factor_race.py --top 5 --cost 0.35

17번에서 복합 점수의 계좌 성과가 벤치마크에 못 미친다는 결과가 나왔다.
그렇다면 **더 단순한 것이 더 낫지 않은가**를 같은 잣대로 확인한다.

특히 [[타인백테스트_검토]] 4단계 D의 논리를 우리에게 적용한다. 그 전략은
상한가 종목을 빼자 엣지가 통째로 사라졌다 — 나머지 조건은 장식이었다.
우리 복합 점수에서 **밸류에이션 부문을 빼면 무엇이 남는가**가 같은 질문이다.

모든 후보는 동일 조건에서 경주한다.
    실전 유니버스(시총 5천억+·거래대금 50억+·시총 상위 200) · 매월 상위 N 동일가중
    편도 거래비용 0.35% · 게이트 없음(순수 순위 능력만 비교)
"""
from __future__ import annotations

import argparse
import importlib.util
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

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "bt17", os.path.join(_here, "17_backtest.py"))
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)


def _pct(s: pd.Series, asof: pd.Series, higher_better: bool) -> pd.Series:
    """월별 백분위 0~100, 높을수록 매수 상위.

    rank(ascending=False)로 뒤집으면 안 된다 — 그러면 결측·동점 처리가
    달라지고, 무엇보다 방향을 헷갈리기 쉽다. grading._percentile과 같은
    방식으로 100에서 빼서 뒤집는다.
    """
    v = pd.to_numeric(s, errors="coerce")
    r = v.groupby(asof).rank(pct=True) * 100
    return r if higher_better else 100 - r


def signals(d: pd.DataFrame) -> dict[str, pd.Series]:
    """후보 신호 → 높을수록 매수 상위."""
    a = d["asof"]
    per = pd.to_numeric(d["per"], errors="coerce").where(lambda x: x > 0)
    pbr = pd.to_numeric(d["pbr"], errors="coerce").where(lambda x: x > 0)

    out = {
        "PBR 저평가 단독": _pct(pbr, a, higher_better=False),
        "PER 저평가 단독": _pct(per, a, higher_better=False),
        "밸류에이션 부문": pd.to_numeric(d["valuation_score"], errors="coerce"),
        "수익성 부문": pd.to_numeric(d["profitability_score"], errors="coerce"),
        "성장성 부문": pd.to_numeric(d["growth_score"], errors="coerce"),
        "안정성 부문": pd.to_numeric(d["stability_score"], errors="coerce"),
    }

    # 복합 점수와, 거기서 밸류에이션만 뺀 버전
    for strat in (strategies.VALUE, strategies.GROWTH):
        w = dict(strategies.WEIGHTS[strat])
        lab = strategies.LABEL[strat]
        out[f"복합: {lab}"] = _weighted(d, w)
        w2 = {k: v for k, v in w.items() if k != "valuation"}
        out[f"복합: {lab} − 밸류에이션"] = _weighted(d, w2)
    return out


def _weighted(d: pd.DataFrame, w: dict[str, float]) -> pd.Series:
    num = pd.Series(0.0, index=d.index)
    den = pd.Series(0.0, index=d.index)
    for cat, weight in w.items():
        if not weight:
            continue
        s = pd.to_numeric(d[f"{cat}_score"], errors="coerce")
        ok = s.notna()
        num = num + s.fillna(0) * weight * ok
        den = den + weight * ok
    return pd.Series(np.where(den > 0, num / den, np.nan), index=d.index)


def run(d: pd.DataFrame, sig: pd.Series, rets, asofs, top, cost):
    x = d.copy()
    x["score"] = sig
    x = x[x["score"].notna()]
    return bt.simulate(x, rets, top, cost, False, asofs)


def summarize(m: pd.DataFrame, bench: np.ndarray, yrs: float) -> dict:
    tot = (m["equity"].iloc[-1] - 1) * 100
    exc = m["ret"].values - bench
    t, p = bt.tstat(pd.Series(exc))
    return {"tot": tot, "cagr": ((1 + tot / 100) ** (1 / yrs) - 1) * 100,
            "mdd": bt.mdd(m["equity"]), "win": (m["ret"] > 0).mean() * 100,
            "exc": exc.mean(), "t": t, "p": p}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--cost", type=float, default=0.35)
    args = ap.parse_args()

    gp = btdata.prepare()
    asofs = sorted(gp["asof"].unique())
    print("월간 수익률 계산 중...")
    rets = btdata.monthly_returns(asofs)
    bm = bt.bench_monthly(gp, rets, asofs)
    yrs = len(asofs) / 12
    bench = bm["eq"].values

    # 신호는 전략 구분과 무관하므로 한쪽 슬라이스만 쓴다
    d = gp[(gp["strategy"] == strategies.VALUE) & gp["in_universe"]].copy()
    sigs = signals(d)

    print("\n" + "=" * 96)
    print(f"전 종목 경주 — 매월 상위 {args.top}종목 동일가중 · 편도비용 {args.cost}%"
          f" · 게이트 없음 ({yrs:.1f}년)")
    print("=" * 96)
    print(f"  {'신호':<26}{'총수익':>10}{'CAGR':>8}{'MDD':>9}{'월승률':>8}"
          f"{'초과':>8}{'t':>7}{'p':>7}")

    rows = {}
    for nm, s in sigs.items():
        m, legs = run(d, s, rets, asofs, args.top, args.cost)
        rows[nm] = (summarize(m, bench, yrs), m, legs)

    eq_tot = ((1 + bm["eq"] / 100).prod() - 1) * 100
    cw_tot = ((1 + bm["cw"] / 100).prod() - 1) * 100
    for nm, lab in ((eq_tot, "[벤치] 유니버스 동일가중"),
                    (cw_tot, "[벤치] 시총가중(지수근사)")):
        col = "eq" if "동일" in lab else "cw"
        e = (1 + bm[col] / 100).cumprod()
        print(f"  {lab:<26}{nm:>+9.1f}%{((1 + nm / 100) ** (1 / yrs) - 1) * 100:>+7.2f}%"
              f"{bt.mdd(e):>8.2f}%{(bm[col] > 0).mean() * 100:>7.1f}%"
              f"{'—':>8}{'—':>7}{'—':>7}")
    print("  " + "-" * 88)
    for nm, (r, _, _) in sorted(rows.items(), key=lambda kv: -kv[1][0]["tot"]):
        print(f"  {nm:<26}{r['tot']:>+9.1f}%{r['cagr']:>+7.2f}%{r['mdd']:>8.2f}%"
              f"{r['win']:>7.1f}%{r['exc']:>7.2f}%{r['t']:>7.2f}{r['p']:>7.3f}")

    # ── 기간 분할 ────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("기간 분할 — 2024~2025 두 해에 의존하는가")
    print("=" * 96)
    yr = np.array([a[:4] for a in asofs])
    early = np.isin(yr, [str(y) for y in range(2018, 2024)])
    late = ~early
    be = ((1 + bm["eq"].values[early] / 100).prod() - 1) * 100
    bl = ((1 + bm["eq"].values[late] / 100).prod() - 1) * 100
    print(f"  {'신호':<26}{'2018~2023':>16}{'2024~2026':>16}{'두 구간 모두 +':>16}")
    print(f"  {'[벤치] 동일가중':<26}{be:>+15.0f}%{bl:>+15.0f}%")
    print("  " + "-" * 88)
    for nm, (_, m, _) in sorted(rows.items(),
                                key=lambda kv: -kv[1][1]["ret"].values[early].mean()):
        e = ((1 + m["ret"].values[early] / 100).prod() - 1) * 100
        l = ((1 + m["ret"].values[late] / 100).prod() - 1) * 100
        both = "O" if (e > be and l > bl) else ("전반만" if e > be else
                                                "후반만" if l > bl else "—")
        print(f"  {nm:<26}{e:>+15.0f}%{l:>+15.0f}%{both:>14}")

    # ── 보유종목 수 민감도 ───────────────────────────────────────────
    print("\n" + "=" * 96)
    print("견고성 — 보유종목 수를 바꾸면 (총수익률)")
    print("=" * 96)
    print(f"  {'신호':<26}{'top3':>12}{'top5':>12}{'top10':>12}{'top20':>12}{'변동폭':>10}")
    for nm, s in sigs.items():
        vals = []
        for n in (3, 5, 10, 20):
            m, _ = run(d, s, rets, asofs, n, args.cost)
            vals.append((m["equity"].iloc[-1] - 1) * 100)
        rng = max(vals) - min(vals)
        print(f"  {nm:<26}" + "".join(f"{v:>+11.0f}%" for v in vals)
              + f"{rng:>9.0f}%p")

    # ── 집중도 ──────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("소수 종목 의존도 — 상위 기여 5종목을 빼면")
    print("=" * 96)
    u = pd.concat([store.load("universe_history_2018_2022"),
                   store.load("universe_20260905")])
    nmap = u.drop_duplicates("ticker").set_index("ticker")["name"].to_dict()
    print(f"  {'신호':<26}{'건당평균':>10}{'상위5 제외':>12}{'변화':>10}  기여 상위 3종목")
    for nm, (_, _, legs) in rows.items():
        if legs.empty:
            continue
        legs = legs.copy()
        legs["nm"] = legs["ticker"].map(nmap).fillna(legs["ticker"])
        c = legs.groupby("nm")["ret"].sum().sort_values(ascending=False)
        base = legs["ret"].mean()
        rest = legs[~legs["nm"].isin(c.head(5).index)]["ret"].mean()
        top3 = " · ".join(list(c.head(3).index))
        print(f"  {nm:<26}{base:>+9.2f}%{rest:>+11.2f}%{rest - base:>+9.2f}%  {top3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
