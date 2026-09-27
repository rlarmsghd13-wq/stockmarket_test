"""데이터 수정 전후 비교 — 같은 규칙으로 옛 패널과 새 패널을 잰다.

    python scripts/27_compare_rebuild.py
    python scripts/27_compare_rebuild.py --old data/curated/_backup_20260926

왜 필요한가
    데이터를 고치면 성과가 움직인다. 그런데 "고쳤더니 좋아졌다"는 말은
    비교 규칙이 같을 때만 뜻이 있다. 2026-09-10에 시세 결손을 고쳤을 때
    벤치마크가 +211% → +103%로 반토막 났는데, 그때는 전후를 나란히 놓고
    본 기록이 없어 어느 숫자가 어느 데이터에서 나온 건지 되짚기 어려웠다.

    유니버스 조건·벤치마크·거래비용은 모두 btdata.prepare()와 17_backtest의
    함수를 그대로 쓴다. 여기서 다시 적으면 비교가 성립하지 않는다.
"""
from __future__ import annotations

import argparse
import importlib.util
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
from src import btdata, store, strategies                # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "d17", os.path.join(os.path.dirname(os.path.abspath(__file__)), "17_backtest.py"))
d17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(d17)

TOP, COST = 5, 0.35


def load(path: str | None) -> pd.DataFrame:
    if path is None:
        return store.load("gate_panel")
    p = os.path.join(path, "gate_panel.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} 가 없습니다.")
    return pd.read_parquet(p)


def summarize(gp: pd.DataFrame, rets: pd.DataFrame, asofs: list[str]) -> dict:
    bm = d17.bench_monthly(gp, rets, asofs)
    out = {}
    for nm, col in (("유니버스 동일가중", "eq"), ("시총가중(지수근사)", "cw")):
        eq = (1 + bm[col] / 100).cumprod()
        out[nm] = (float((eq.iloc[-1] - 1) * 100), d17.mdd(eq))
    picks = {}
    for strat in (strategies.VALUE, strategies.GROWTH):
        d = gp[(gp["strategy"] == strat) & gp["in_universe"]].copy()
        for gate in (True, False):
            m, _ = d17.simulate(d, rets, TOP, COST, gate, asofs)
            eq = (1 + m["ret"] / 100).cumprod()
            lab = f"{strategies.LABEL[strat]} 게이트{'O' if gate else 'X'}"
            out[lab] = (float((eq.iloc[-1] - 1) * 100), d17.mdd(eq))
            sub = d[d["pass"]] if gate else d
            picks[lab] = {a: set((sub[sub["asof"] == a]
                                 .sort_values("score", ascending=False)
                                 .head(TOP)["ticker"]))
                          for a in asofs}
    return {"perf": out, "picks": picks, "n_uni": gp[gp["in_universe"]]
            .groupby("asof")["ticker"].nunique()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=str(config.CURATED / "_backup_20260926"))
    args = ap.parse_args()

    new_gp_raw, old_gp_raw = load(None), load(args.old)
    # 두 패널의 공통 기준일만 본다. 기간이 다르면 수익률 비교가 성립하지 않는다.
    common = sorted(set(new_gp_raw["asof"]) & set(old_gp_raw["asof"]))
    print(f"옛 패널 {old_gp_raw['asof'].nunique()}개월 · "
          f"새 패널 {new_gp_raw['asof'].nunique()}개월 · 공통 {len(common)}개월")

    print("\n[옛 패널]")
    old = btdata.prepare(old_gp_raw[old_gp_raw["asof"].isin(common)])
    print("\n[새 패널]")
    new = btdata.prepare(new_gp_raw[new_gp_raw["asof"].isin(common)])

    print("\n월간 수익률 계산 중...")
    rets = btdata.monthly_returns(common)
    o = summarize(old, rets, common)
    n = summarize(new, rets, common)

    print("\n" + "=" * 86)
    print(f"① 실전 유니버스 종목 수 (월별, {len(common)}개월)")
    print("=" * 86)
    print(f"  {'':<10}{'평균':>8}{'최소':>8}{'최대':>8}")
    for lab, s in (("옛 패널", o["n_uni"]), ("새 패널", n["n_uni"])):
        print(f"  {lab:<10}{s.mean():>8.0f}{s.min():>8.0f}{s.max():>8.0f}")

    print("\n" + "=" * 86)
    print("② 총수익 / MDD — 같은 유니버스 조건·같은 거래비용")
    print("=" * 86)
    print(f"  {'':<22}{'옛 총수익':>12}{'새 총수익':>12}{'변화':>11}"
          f"{'옛 MDD':>10}{'새 MDD':>10}")
    for k in o["perf"]:
        (ot, om), (nt, nm) = o["perf"][k], n["perf"][k]
        print(f"  {k:<22}{ot:>+11.1f}%{nt:>+11.1f}%{nt-ot:>+10.1f}%p"
              f"{om:>9.1f}%{nm:>9.1f}%")

    print("\n" + "=" * 86)
    print(f"③ 매수 종목이 바뀐 달 — 상위 {TOP}종목이 하나라도 다르면 '바뀜'")
    print("=" * 86)
    print(f"  {'':<22}{'바뀐 달':>9}{'비율':>8}{'평균 교체':>11}")
    for k in o["picks"]:
        ch, repl = 0, []
        for a in common:
            a1, b1 = o["picks"][k].get(a, set()), n["picks"][k].get(a, set())
            if a1 != b1:
                ch += 1
            if a1 or b1:
                repl.append(len(b1 - a1))
        avg = sum(repl) / len(repl) if repl else 0
        print(f"  {k:<22}{ch:>9}{ch/len(common)*100:>7.0f}%{avg:>10.1f}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
