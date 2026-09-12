"""게이트 후속 — 실전 유니버스 조건에서, 올바른 벤치마크로 다시 본다.

    python scripts/16_gate_followup.py

15번에서 "게이트를 끄면 좋아진다"가 나왔지만 그 이득은 시총 Q1(중앙 2,768억)에서
나왔다. 실전 유니버스(src/universe.py:68)는 이미 시총 5천억 · 거래대금 50억 ·
시총 상위 200으로 자른다. **닿을 수 없는 수익을 근거로 게이트를 판단하면 안 된다.**

벤치마크
    14번은 초과수익을 전체 패널(소형주 포함) 평균 대비로 쟀다. 소형주가 크게
    앞섰으므로 대형주만 담는 전략은 자동으로 불리해진다. 여기서는 세 가지를 나눈다.
      · 전체 패널 동일가중 — 참고용
      · **실전 유니버스 동일가중** — 실제로 대체 가능한 선택지. 이게 기준이다
      · 실전 유니버스 시총가중 — 지수 근사

    ① 실전 유니버스 조건에서 게이트 재평가
    ② 시장 국면별 게이트 성과 — 방어 장치 가설 검정
    ③ 레짐 지표의 정체 — 게이트 통과율인가, 매출성장 확산지수인가
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

import config  # noqa: E402
from src import store, strategies  # noqa: E402

TOP_N = 5


def cluster_t(g: pd.DataFrame, col: str) -> tuple[float, float, int]:
    m = g.groupby("asof")[col].mean().dropna()
    if len(m) < 3 or m.std(ddof=1) == 0:
        return float("nan"), float("nan"), len(m)
    t = m.mean() / (m.std(ddof=1) / math.sqrt(len(m)))
    return float(t), float(math.erfc(abs(t) / math.sqrt(2))), len(m)


def picks(d: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    return d.sort_values("score", ascending=False).groupby("asof").head(n)


HEAD = f"  {'':<20}{'월수':>5}{'평균':>10}{'초과':>10}{'승률':>8}{'t':>7}{'p':>8}"


def line(nm: str, p: pd.DataFrame, col: str = "exc_univ") -> str:
    t, pv, n = cluster_t(p, col)
    return (f"  {nm:<20}{n:>5}{p['fwd_63'].mean():>9.2f}%"
            f"{p[col].mean():>9.2f}%"
            f"{(p['fwd_63'] > 0).mean() * 100:>7.1f}%{t:>7.2f}{pv:>8.3f}")


def prepare() -> pd.DataFrame:
    gp = store.load("gate_panel")
    px = pd.concat([store.load("prices_daily_hist"), store.load("prices_daily")],
                   ignore_index=True).drop_duplicates(["ticker", "date"], keep="last")
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])
    px["tv"] = px["close_adj"] * px["volume"]
    px["avg_tv"] = (px.groupby("ticker")["tv"]
                      .transform(lambda s: s.rolling(60, min_periods=20).mean()))
    gp["asof_ts"] = pd.to_datetime(gp["asof"])
    gp = pd.merge_asof(gp.sort_values("asof_ts"),
                       px[["ticker", "date", "avg_tv"]].dropna().sort_values("date"),
                       left_on="asof_ts", right_on="date", by="ticker",
                       direction="backward")

    gp["cap_rank"] = gp.groupby(["asof", "strategy"])["market_cap"].rank(
        ascending=False, method="min")
    gp["in_universe"] = (
        (gp["market_cap"] >= config.MIN_MARKET_CAP)
        & (gp["avg_tv"].fillna(0) >= config.MIN_AVG_TRADING_VALUE)
        & (gp["cap_rank"] <= config.UNIVERSE_SIZE))

    # 벤치마크 세 가지
    u = gp[gp["in_universe"]]
    bench_eq = u.groupby(["asof", "strategy"])["fwd_63"].mean().rename("b_univ")
    cw = (u.assign(w=u["market_cap"])
           .groupby(["asof", "strategy"])
           .apply(lambda g: np.average(g["fwd_63"], weights=g["w"]),
                  include_groups=False).rename("b_cap"))
    gp = gp.merge(bench_eq, on=["asof", "strategy"], how="left")
    gp = gp.merge(cw, on=["asof", "strategy"], how="left")
    gp["exc_univ"] = gp["fwd_63"] - gp["b_univ"]
    gp["exc_cap"] = gp["fwd_63"] - gp["b_cap"]
    return gp


def main() -> int:
    gp = prepare()
    v = gp[gp["strategy"] == strategies.VALUE]
    print(f"패널 {len(v):,}행 → 실전 유니버스 통과 {int(v['in_universe'].sum()):,}행"
          f" ({v['in_universe'].mean() * 100:.1f}%) · 월평균 "
          f"{v.groupby('asof')['in_universe'].sum().mean():.0f}종목")
    b = v.groupby("asof")[["mkt", "b_univ", "b_cap"]].first()
    print(f"  벤치마크 3개월 평균수익률 — 전체패널 {b['mkt'].mean():+.2f}%"
          f" · 실전유니버스 동일가중 {b['b_univ'].mean():+.2f}%"
          f" · 시총가중(지수근사) {b['b_cap'].mean():+.2f}%")

    # ── ① 실전 유니버스에서 게이트 재평가 ────────────────────────────
    print("\n" + "=" * 84)
    print(f"① 실전 유니버스 안에서 상위 {TOP_N}종목 — 유니버스 동일가중 대비")
    print("=" * 84)
    for strat in (strategies.VALUE, strategies.GROWTH):
        d = gp[(gp["strategy"] == strat) & gp["in_universe"]].copy()
        print(f"\n[{strategies.LABEL[strat]}]")
        print(HEAD)
        print(line("게이트 켬", picks(d[d["pass"]])))
        print(line("게이트 끔", picks(d)))
        print(line("  (지수근사 대비)", picks(d[d["pass"]]), "exc_cap"))

    # ── ② 시장 국면별 ───────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("② 시장 국면별 — 게이트는 방어 장치인가")
    print("=" * 84)
    for strat in (strategies.VALUE, strategies.GROWTH):
        d = gp[(gp["strategy"] == strat) & gp["in_universe"]].copy()
        mk = d.groupby("asof")["b_univ"].first()
        q = mk.quantile([1 / 3, 2 / 3]).tolist()
        rg = pd.cut(mk, [-999, q[0], q[1], 999],
                    labels=["하락장", "횡보장", "상승장"]).rename("regime")
        d = d.merge(rg, on="asof", how="left")
        print(f"\n[{strategies.LABEL[strat]}]")
        print(f"  {'국면':<10}{'유니버스':>10}{'게이트 켬':>12}{'게이트 끔':>12}"
              f"{'차이':>10}{'월수':>7}")
        for nm in ["하락장", "횡보장", "상승장"]:
            s = d[d["regime"] == nm]
            if s.empty:
                continue
            on = picks(s[s["pass"]])["exc_univ"].mean()
            off = picks(s)["exc_univ"].mean()
            print(f"  {nm:<10}{s['b_univ'].mean():>9.1f}%{on:>11.2f}%"
                  f"{off:>11.2f}%{on - off:>+9.2f}%{s['asof'].nunique():>7}")

    # ── ③ 레짐 지표 ────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("③ 레짐 지표 — 무엇이 이후 시장을 예측하는가 (유니버스 동일가중 기준)")
    print("=" * 84)
    g = gp[gp["strategy"] == strategies.GROWTH].copy()
    g["rev3"] = pd.to_numeric(g["rev_growth_q_of_4"], errors="coerce") >= 3
    gu = g[g["in_universe"]]
    cand = {
        "성장주 게이트 통과율": gu.groupby("asof")["pass"].mean(),
        "매출성장 확산지수(4분기중 3회+)": gu.groupby("asof")["rev3"].mean(),
        "가치 게이트 통과율": gp[(gp["strategy"] == strategies.VALUE)
                            & gp["in_universe"]].groupby("asof")["pass"].mean(),
    }
    mkt = gu.groupby("asof")["b_univ"].first()
    print(f"  {'지표':<30}{'순위상관':>10}{'t':>8}{'p':>9}"
          f"{'하위1/3→시장':>14}{'상위1/3→시장':>14}")
    for nm, s in cand.items():
        c = pd.concat([s.rename("x"), mkt.rename("m")], axis=1).dropna()
        r = c["x"].rank().corr(c["m"].rank())
        n = len(c)
        t = r * math.sqrt((n - 2) / max(1e-9, 1 - r * r))
        p = math.erfc(abs(t) / math.sqrt(2))
        qq = c["x"].quantile([1 / 3, 2 / 3]).tolist()
        lo, hi = c[c["x"] <= qq[0]]["m"].mean(), c[c["x"] >= qq[1]]["m"].mean()
        print(f"  {nm:<30}{r:>+10.3f}{t:>8.2f}{p:>9.3f}{lo:>13.2f}%{hi:>13.2f}%")

    s = cand["매출성장 확산지수(4분기중 3회+)"]
    c = pd.concat([s.rename("x"), mkt.rename("m")], axis=1).dropna()
    half = len(c) // 2
    print("\n  매출성장 확산지수 — 기간 분할")
    for nm, sub in (("전반기", c.iloc[:half]), ("후반기", c.iloc[half:])):
        r = sub["x"].rank().corr(sub["m"].rank())
        print(f"    {nm} ({sub.index[0]} ~ {sub.index[-1]})  순위상관 {r:+.3f}"
              f"  (월 {len(sub)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
