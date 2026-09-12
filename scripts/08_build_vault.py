"""옵시디언 볼트 생성 — 월 1회 실행.

    python scripts/08_build_vault.py --asof 2026-09-05

기존 노트가 있으면 월별 이력을 이어 붙이고, 전략이 바뀐 종목은 폴더를 옮기며
변동 이력을 남긴다.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import config  # noqa: E402
from src import consensus, env, regime, report, sectors, store, strategies, vault  # noqa: E402

README = """---
tags: [안내]
---

# 주식 스크리닝 볼트

월 1회 `screener/scripts/08_build_vault.py`가 생성합니다.

## 구조

- **월간리포트/** — 그 달의 상위 업종 5개와 전략별 상위 5종목
- **가치투자/ 성장주/** — 종목당 노트 1개. 전략이 바뀌면 폴더가 바뀝니다
- **_성과추적/** — 후보 규칙들의 선택을 매달 기록·정산 (아웃오브샘플)
- **_참고/** — 백테스트 결과와 검증 방법론
- 종목 노트에는 **월별 이력이 누적**되므로 점수 추이를 한 파일에서 볼 수 있습니다

## 점수 읽는 법

점수는 **같은 트랙·같은 업종 안에서의 백분위**를 부문별로 가중평균한 값입니다.
절대 수치가 아니라 동종 비교군 내 순위입니다.

- 90+ `A+` · 75+ `A` · 45+ `B` · 20+ `C` · 그 미만 `D`
- **밴드 위치** — 그 종목의 최근 2년 PER 분포에서 지금이 몇 번째 백분위인지.
  낮을수록 자기 과거 대비 싼 구간입니다

## 주의

8년 계좌 백테스트에서 가치투자는 벤치마크에 졌고, 성장주는 이겼지만 낙폭이
지수의 두 배였습니다. 자세한 내용은 `_참고/백테스트_2026-09.md`.
**정렬 기준으로 쓰고 예측값으로 쓰지 마세요.**

증권사 목표가는 뉴스 인용을 모은 **참고 정보**이며 점수에 들어가지 않습니다.
목표가는 체계적으로 낙관적이라 절대값보다 종목 간 순위와 상향·하향 방향을 보세요.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=date.today().isoformat())
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--vault", default=None)
    args = ap.parse_args()

    env.load()          # .env 의 VAULT_ROOT 를 반영한 뒤에 경로를 읽는다
    root = Path(args.vault) if args.vault else Path(config.VAULT_ROOT)
    month = args.asof[:7]

    g = store.load("grades_latest")
    sc = store.load("strategy_scores")
    px = store.load("prices_daily")
    ttm = store.load("fin_ttm")
    tracks = store.load("tracks").set_index("ticker")

    for sub in ["월간리포트", *vault.FOLDER.values()]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    consensus.ensure(root)
    cons_df = consensus.load(root)
    if not cons_df.empty:
        print(f"  컨센서스 입력 {len(cons_df)}건 · {cons_df.ticker.nunique()}종목")
    # 뉴스 인용 목표가 (21_news_consensus.py) — 있으면 수동 입력보다 우선한다
    news_obs, news_cons, news_sum = pd.DataFrame(), pd.DataFrame(), {}
    if store.exists("news_targets") and store.exists("news_consensus"):
        news_obs = store.load("news_targets")
        news_obs["pub"] = pd.to_datetime(news_obs["pub"], utc=True)
        news_cons = store.load("news_consensus")
        print(f"  뉴스 컨센서스 {int((news_cons['n_brokers'] >= 3).sum())}종목"
              f" (수집 {news_cons['asof'].iloc[0]})")
    asof_end = pd.Timestamp(args.asof).tz_localize("Asia/Seoul") + pd.Timedelta(hours=23, minutes=59)

    # ── 종목별 귀속 전략: 게이트를 통과한 전략 중 최고 점수 ──────────────
    passed = sc[sc.gate_pass].copy()
    best = passed.loc[passed.groupby("ticker")["score"].idxmax()]
    gate_counts = {s: int(((sc.strategy == s) & sc.gate_pass).sum())
                   for s in vault.FOLDER}

    # 히스테리시스를 적용하려면 "지금 들고 있는가"를 알아야 한다.
    # 별도 포트폴리오 파일 없이 기존 노트의 직전 월 판정에서 읽는다.
    held: dict[str, bool] = {}
    prev_cache: dict[str, tuple] = {}
    for t in best.ticker:
        old = vault.find_existing(root, t)
        if old is None:
            held[t] = False
            continue
        fm, secs = vault.read_note(old)
        rows = vault.parse_rows(secs.get("월별 이력", ""))
        prev_cache[t] = (old, fm, rows, vault.parse_rows(secs.get("전략 변동 이력", "")))
        held[t] = vault.was_held(rows)

    # ── 직전 스냅샷과 비교해 시장 국면(폭)을 잰다 ────────────────────────
    prev_g = None
    snaps = sorted(p.stem for p in (config.CURATED).glob("grades_2*.parquet"))
    cur_tag = f"grades_{args.asof.replace('-', '')}"
    older = [s for s in snaps if s < cur_tag]
    if older:
        prev_g = store.load(older[-1])
        print(f"  직전 스냅샷: {older[-1]}")
    bd = regime.breadth(prev_g, g)
    if bd["n"]:
        print(f"  시장 국면: 전체 {bd['overall']*100:.0f}% 동반 악화 (표본 {bd['n']}종목)")
        for tr, v in sorted(bd["track"].items()):
            mark = " ← 시장 요인" if v >= regime.BREADTH_MARKET else ""
            print(f"    {tr} {v*100:>3.0f}%{mark}")
    prev_rows_by_t = ({r.ticker: r._asdict() for r in prev_g.itertuples()}
                      if prev_g is not None else {})

    picks: dict[str, list[dict]] = {}
    for s in vault.FOLDER:
        sel = best[best.strategy == s].nlargest(args.top, "score")
        picks[s] = [
            report.build(t, g, sc, px, ttm, s, args.asof, held.get(t, False),
                         prev_row=prev_rows_by_t.get(t), bd=bd,
                         defer_months=vault.defer_months(
                             prev_cache[t][2] if t in prev_cache else []))
            for t in sel.ticker]

    # ── 노트 쓰기 ────────────────────────────────────────────────────────
    moves, written = [], 0
    for s, rs in picks.items():
        for r in rs:
            if not r:
                continue
            sec = sectors.name(tracks.loc[r["ticker"], "induty_code"]
                               if r["ticker"] in tracks.index else None)
            folder = root / vault.FOLDER[s]
            target = folder / f"{vault.safe(r['name'])}.md"

            cached = prev_cache.get(r["ticker"])
            prev_rows, prev_moves, moved_from = [], [], None
            if cached is not None:
                old, fm, prev_rows, prev_moves = cached
                if old.exists() and old.resolve() != target.resolve():
                    moved_from = fm.get("현재전략") or old.parent.name
                    old.unlink()          # 새 폴더에 다시 쓰므로 원본은 지운다

            ns = consensus.news_summary(news_obs, news_cons, r["ticker"], r.get("price"), asof_end)
            news_sum[r["ticker"]] = ns
            cs = consensus.summarize(cons_df, r["ticker"], r.get("price"))
            parts = []
            if ns:
                parts.append(consensus.news_markdown(ns))
            if cs and cs.get("n"):
                parts.append("### 직접 입력한 리포트\n\n" + consensus.markdown(cs))
            text = vault.stock_note(r, sec, month, prev_rows, prev_moves, moved_from,
                                    "\n\n".join(parts))
            target.write_text(text, encoding="utf-8")
            written += 1
            if moved_from:
                line = (f"| {month} | [[{vault.safe(r['name'])}]] "
                        f"{moved_from} → **{vault.FOLDER[s]}** | {r['score']:.1f} | "
                        f"전략 귀속 변경 |")
                moves.append(line)
                print(f"  [이동] {r['name']}: {moved_from} → {vault.FOLDER[s]}")

    # ── 업종 순위 ────────────────────────────────────────────────────────
    bt = best[["ticker", "score"]].merge(
        g[["ticker", "name"]], on="ticker", how="left")
    bt["sector_name"] = [sectors.name(tracks.loc[t, "induty_code"])
                         if t in tracks.index else "미분류" for t in bt.ticker]
    grp = []
    for sn, sub in bt.groupby("sector_name"):
        if len(sub) < 3:
            continue
        top = sub.loc[sub.score.idxmax()]
        grp.append({"sector_name": sn, "n": len(sub),
                    "mean_score": sub.score.mean(),
                    "top_name": top["name"], "top_score": top.score})
    sec_df = pd.DataFrame(grp).nlargest(5, "mean_score") if grp else pd.DataFrame()

    # ── 월간 노트 ────────────────────────────────────────────────────────
    (root / "README.md").write_text(README, encoding="utf-8")
    # 수동 조회 목록에는 뉴스로 증권사 3곳 이상이 잡힌 종목을 뺀다
    have = set(cons_df["ticker"]) if not cons_df.empty else set()
    have |= {t for t, s in news_sum.items() if s and s.get("n", 0) >= 3}
    todo = consensus.todo_block(
        {vault.FOLDER[s]: rs for s, rs in picks.items()}, args.asof, have)
    all_picks = [r for rs in picks.values() for r in rs if r]
    usable = {t: s for t, s in news_sum.items() if s and s.get("target_median")}
    compare_md = consensus.compare_markdown(consensus.compare(all_picks, cons_df, usable))
    mn = vault.month_note(month, args.asof, sec_df, picks, moves,
                          int(g.ticker.nunique()), gate_counts, todo, compare_md)
    (root / "월간리포트" / f"{month}.md").write_text(mn, encoding="utf-8")

    print(f"\n볼트: {root}")
    print(f"  월간리포트/{month}.md")
    for s, label in vault.FOLDER.items():
        n = len(list((root / label).glob('*.md')))
        print(f"  {label}/ — {n}개 노트 (이번 달 상위 {len(picks[s])})")
    print(f"  종목 노트 {written}개 갱신 · 전략 이동 {len(moves)}건")
    if not sec_df.empty:
        print("\n상위 업종 5:")
        for i, r in enumerate(sec_df.itertuples(), 1):
            print(f"  {i}. {r.sector_name:<14} {r.n}종목  평균 {r.mean_score:.1f}"
                  f"  최고 {r.top_name} {r.top_score:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
