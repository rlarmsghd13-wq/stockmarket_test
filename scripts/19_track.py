"""성과 추적 — 매달 후보 규칙의 선택을 기록하고 직전 달을 정산한다.

    python scripts/19_track.py --asof 2026-09-09

17·18번 백테스트는 전부 인샘플이다. 이 스크립트는 **오늘부터** 오염되지 않은
표본을 쌓는다. 실제 매수는 복합 점수로 하되, 성장성·수익성·밸류에이션·PBR
단독 규칙도 종이 위에서 같이 굴려 1년 뒤 기록으로 비교한다.

run_monthly.py가 마지막 단계로 호출한다. 단독 실행도 가능하다.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import config  # noqa: E402
from src import env, store, strategies, track  # noqa: E402

HEADER = """---
tags: [성과추적]
갱신일: {today}
추적시작: {start}
---

# 성과 추적 (아웃오브샘플)

17·18번 백테스트는 **인샘플**이었다 — 가중치를 그 기간으로 맞췄고 12개 신호를
비교해 제일 좋은 것을 골랐다. 그래서 결론에 손대지 않고, 대신 **{start}부터**
후보 규칙 6개의 선택을 매달 기록한다.

실제 매수는 `복합:가치투자` · `복합:성장주`(게이트 적용)로 하고,
나머지 4개는 비교용 종이 기록이다.

> **{months}개월치 기록입니다.** 최소 12개월은 쌓여야 방향을 논할 수 있고,
> 통계적 판단에는 그보다 훨씬 더 필요합니다. 지금 순위로 규칙을 바꾸지 마세요.
"""


def markdown(root: Path, asof: str) -> str:
    mon = track.load_csv(root, track.MONTHLY, track.MONTHLY_COLS)
    led = track.load_csv(root, track.LEDGER, track.LEDGER_COLS)
    start = led["asof"].min() if not led.empty else asof
    out = [HEADER.format(today=asof, start=start, months=len(mon))]

    if mon.empty:
        out.append("\n아직 정산된 달이 없습니다. 다음 달 실행 때 첫 결과가 나옵니다.\n")
    else:
        out.append("\n## 누적 성과\n")
        out.append("| 규칙 | 누적 | 월평균 | 벤치 대비 | 이긴 달 | 최고 | 최악 |")
        out.append("|---|---|---|---|---|---|---|")
        be = (1 + mon["bench_eq"] / 100).prod() - 1
        bc = (1 + mon["bench_cap"] / 100).prod() - 1
        rows = []
        for rule in track.RULE_ORDER:
            v = pd.to_numeric(mon[rule], errors="coerce").dropna()
            if v.empty:
                continue
            cum = (1 + v / 100).prod() - 1
            d = v.values - mon.loc[v.index, "bench_eq"].values
            rows.append((cum, rule, v, d))
        for cum, rule, v, d in sorted(rows, reverse=True):
            out.append(f"| {rule} | {cum * 100:+.1f}% | {v.mean():+.2f}% | "
                       f"{d.mean():+.2f}%p | {(d > 0).sum()}/{len(d)} | "
                       f"{v.max():+.1f}% | {v.min():+.1f}% |")
        out.append(f"| *벤치마크 동일가중* | {be * 100:+.1f}% | "
                   f"{mon['bench_eq'].mean():+.2f}% | — | — | "
                   f"{mon['bench_eq'].max():+.1f}% | {mon['bench_eq'].min():+.1f}% |")
        out.append(f"| *벤치마크 시총가중* | {bc * 100:+.1f}% | "
                   f"{mon['bench_cap'].mean():+.2f}% | — | — | "
                   f"{mon['bench_cap'].max():+.1f}% | {mon['bench_cap'].min():+.1f}% |")

        out.append("\n## 월별\n")
        cols = ["asof", "days", "bench_eq", "bench_cap", "diffusion"] + track.RULE_ORDER
        head = "| 기준일 | 일수 | 벤치(동일) | 벤치(시총) | 확산지수 | " + \
               " | ".join(track.RULE_ORDER) + " |"
        out.append(head)
        out.append("|" + "---|" * len(cols))
        for r in mon.sort_values("asof", ascending=False).itertuples():
            cells = [str(r.asof), f"{r.days:.0f}",
                     f"{r.bench_eq:+.2f}%", f"{r.bench_cap:+.2f}%",
                     f"{r.diffusion:.1f}%" if pd.notna(r.diffusion) else "—"]
            for rule in track.RULE_ORDER:
                v = getattr(r, rule.replace(":", "_").replace(" ", "_"), None)
                v = mon.loc[mon["asof"] == r.asof, rule].iloc[0]
                cells.append(f"{v:+.2f}%" if pd.notna(v) else "—")
            out.append("| " + " | ".join(cells) + " |")

        dif = mon["diffusion"].dropna()
        if len(dif) >= 2:
            out.append(f"\n## 매출성장 확산지수\n")
            out.append(f"최근 {dif.iloc[-1]:.1f}% · 기록 중앙값 {dif.median():.1f}%")
            out.append("\n검증(97개월)에서 이후 3개월 시장수익률과 순위상관 +0.48이었다. "
                       "하위 1/3 구간의 이후 시장은 −4.3%, 상위 1/3은 +10.9%.")
            out.append("**아직 매매 근거가 아니다 — 기록해서 확인하는 중.**")

    # 이번 달 선택
    cur = led[led["asof"] == asof]
    if not cur.empty:
        out.append(f"\n## {asof} 선택\n")
        for rule in track.RULE_ORDER:
            g = cur[cur["rule"] == rule].sort_values("rank")
            if g.empty:
                out.append(f"- **{rule}** — 조건 통과 종목 없음 (현금)")
                continue
            names = " · ".join(f"{r.name}({r.score:.0f})" for r in g.itertuples())
            out.append(f"- **{rule}** — {names}")

    out.append("\n## 관련\n")
    out.append("- [[백테스트_2026-09]] — 인샘플 결과와 그 한계")
    out.append("- [[타인백테스트_검토]] — 검증 표 구성의 출처")
    out.append("- 원장: `_성과추적/원장.csv` · 월별: `_성과추적/월별성과.csv`")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--top", type=int, default=track.TOP_N)
    ap.add_argument("--vault", default=None)
    args = ap.parse_args()

    env.load()          # .env 의 VAULT_ROOT 를 반영한 뒤에 경로를 읽는다
    root = Path(args.vault) if args.vault else Path(config.VAULT_ROOT)
    g = store.load("grades_latest")
    px = store.load("prices_daily")
    ttm = store.load("fin_ttm")

    # 게이트 — 실제 볼트에 올라가는 것과 같은 판정
    gf = strategies.gate_features(ttm, args.asof)
    df = g.merge(gf, on="ticker", how="left")
    # 뉴스 컨센서스 (21_news_consensus.py) — 있으면 컨센서스 규칙 3개가 켜진다
    if store.exists("news_consensus"):
        nc_df = store.load("news_consensus")[["ticker", "n_brokers", "upside", "revision"]]
        df = df.merge(nc_df, on="ticker", how="left")
        print(f"  컨센서스 결합: 증권사 3곳 이상 {int((df['n_brokers'] >= 3).sum())}종목")
    gate_pass = {}
    for s in (strategies.VALUE, strategies.GROWTH):
        r = strategies.score(df, s).set_index("ticker")["gate_pass"]
        gate_pass[s] = df["ticker"].map(r)

    # ① 정산
    n, months = track.settle(root, px, args.asof)
    if n:
        print(f"  정산 {n}건 · 대상 월 {', '.join(months)}")
        for m in months:
            snap_tag = f"grades_{m.replace('-', '')}"
            snap = store.load(snap_tag) if store.exists(snap_tag) else g
            dif = track.diffusion_index(ttm, m, list(snap["ticker"]))
            row = track.close_month(root, m, args.asof, px, snap, dif)
            if row:
                best = max(track.RULE_ORDER,
                           key=lambda k: row.get(k) if pd.notna(row.get(k)) else -999)
                print(f"    {m} → 벤치 {row['bench_eq']:+.2f}%"
                      f" · 최고 {best} {row[best]:+.2f}%")
    else:
        print("  정산할 건이 없습니다 (첫 실행이거나 같은 달 재실행)")

    # ② 이번 달 기록
    sel = track.selections(df, gate_pass, args.top)
    added = track.record(root, args.asof, sel, px)
    print(f"  기록 {added}건 · 규칙 {sel['rule'].nunique()}개")
    for rule in track.RULE_ORDER:
        s = sel[sel["rule"] == rule].sort_values("rank")
        names = " · ".join(s["name"].astype(str)) if not s.empty else "(없음 — 현금)"
        print(f"    {rule:<14} {names}")

    dif = track.diffusion_index(ttm, args.asof, list(g["ticker"]))
    if dif is not None:
        print(f"  매출성장 확산지수 {dif * 100:.1f}%")

    p = root / track.FOLDER / track.NOTE
    p.write_text(markdown(root, args.asof), encoding="utf-8")
    print(f"  → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
