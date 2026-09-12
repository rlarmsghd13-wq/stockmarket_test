"""뉴스 인용 목표주가 → 종목별 컨센서스 (NAVER API HUB).

    python scripts/21_news_consensus.py --test          # 10종목 추출 정확도 확인
    python scripts/21_news_consensus.py                 # 유니버스 전체 (월간 실행에서 호출)
    python scripts/21_news_consensus.py --show 005380   # 증권사별 표

검증 (2026-09-11) — 네이버 증권 컨센서스(평균)와 대조
    KB금융  우리 중간값 220,000 (15곳) vs 225,444  −2.4%
    현대차  우리 중간값 720,000 (23곳) vs 701,600  +2.6%
    기준 ±10% 통과 → 전 종목 확대.

누적 저장
    검색 API는 검색어당 최신 1,000건까지만 준다. 대형주는 그게 약 30일치라
    한 번 받아서는 90일 창을 못 채운다. 기사 단위 관측치를 `news_obs_hist`에
    쌓고, 집계는 쌓인 이력 전체로 한다. 자주 돌릴수록 대형주 커버리지가 는다.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src import env, news_consensus as nc, store  # noqa: E402

TEST = ["005930", "000660", "005380", "105560", "196170",
        "103590", "111770", "062040", "267270", "263750"]
HIST = "news_obs_hist"
BAND = (0.5, 3.0)        # 기사 날짜 주가 대비 목표가 배율 — 밖이면 파싱 오류로 본다


def collect(tickers: list[str], names: dict[str, str], since: datetime) -> tuple[pd.DataFrame, dict]:
    """API로 오늘 분을 받고(캐시), 관측치는 **받아둔 원본 전체**에서 현재 규칙으로 뽑는다."""
    universe = list(names.values())
    rows, span = [], {}
    for i, t in enumerate(tickers, 1):
        name = names[t]
        query = f"{name} 목표주가"
        items = nc.search(query, since=since)
        others = nc.other_patterns(name, universe)
        for it in nc.cached_items(query):
            for o in nc.extract(it, name, others):
                rows.append({"ticker": t, "name": name, **o})
        if items:
            pubs = pd.to_datetime([it["pubDate"] for it in items],
                                  format="%a, %d %b %Y %H:%M:%S %z")
            span[t] = (len(items), int((datetime.now(nc.KST) - pubs.min()).days))
        else:
            span[t] = (0, 0)
        if i % 40 == 0:
            print(f"  {i}/{len(tickers)} 종목 수집")
    d = pd.DataFrame(rows)
    if not d.empty:
        d["pub"] = pd.to_datetime(d["pub"], utc=True)
    return d, span


def save_history(d: pd.DataFrame) -> pd.DataFrame:
    """관측치 스냅샷 (점검용). 기준은 원본 캐시이고 이건 매번 새로 만든다."""
    if d.empty:
        return d
    d["_t"] = d["target"].round()
    d = d.drop_duplicates(["ticker", "link", "broker", "_t"], keep="last").drop(columns="_t")
    store.save(d.assign(pub=d["pub"].astype(str)), HIST)
    return d


def market_names() -> list[str]:
    """전체 상장사명 — "CJ"가 "CJ제일제당" 안에서 잡히지 않게 막는 데 쓴다."""
    names = set()
    for n in ("universe_20260905", "universe_history_2018_2022"):
        if store.exists(n):
            names |= set(store.load(n)["name"].dropna().astype(str))
    return sorted(names)


def band_filter(obs: pd.DataFrame, px: pd.DataFrame, t: str) -> tuple[pd.DataFrame, int]:
    """기사 날짜 주가 대비 0.5배 미만·3배 초과 목표가 제거
    (첫 테스트의 KB금융 280만원·4만원, 삼성전자 3.2만원이 여기 걸린다)."""
    s = px[px["ticker"] == t][["date", "close_adj"]].sort_values("date")
    # 한국 날짜 기준 — UTC로 자르면 새벽 기사가 전날 주가와 비교된다
    o = obs.assign(d=obs["pub"].dt.tz_convert("Asia/Seoul").dt.tz_localize(None).dt.normalize()).sort_values("d")
    o = pd.merge_asof(o, s, left_on="d", right_on="date", direction="backward")
    o["ratio"] = o["target"] / o["close_adj"]
    bad = ~o["ratio"].between(*BAND) & o["ratio"].notna()
    return o[~bad].drop(columns=["d", "date", "close_adj"]), int(bad.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--show", default="", help="증권사별 표를 보여줄 종목코드(쉼표)")
    args = ap.parse_args()
    env.load()

    g = store.load("grades_latest")
    px = store.load("prices_daily")
    px["date"] = pd.to_datetime(px["date"])
    price = px.sort_values("date").groupby("ticker")["close_adj"].last()
    names = dict(zip(g["ticker"], g["name"]))
    tickers = TEST if args.test else list(g["ticker"])

    nc.MARKET_NAMES = market_names()
    now = datetime.now(nc.KST)
    since = now - timedelta(days=nc.WINDOW_DAYS)
    before = nc.api_calls_in_cache(now.strftime("%Y%m%d"))
    print(f"수집 {len(tickers)}종목 · 최근 {nc.WINDOW_DAYS}일 · 상장사명 {len(nc.MARKET_NAMES):,}개")
    new, span = collect(tickers, names, since)
    hist = save_history(new)
    calls = nc.api_calls_in_cache(now.strftime("%Y%m%d")) - before
    print(f"  API 호출 {calls}회 (오늘 누적 {nc.api_calls_in_cache(now.strftime('%Y%m%d'))}회)"
          f" · 이력 {len(hist):,}건")

    rows, kept = [], []
    for t in tickers:
        h = hist[hist["ticker"] == t] if not hist.empty else hist
        n_band = 0
        if not h.empty:
            h = nc.dedupe(h)
            h, n_band = band_filter(h, px, t)
            kept.append(h)
        agg = nc.aggregate(h, float(price.get(t, float("nan"))), now)
        n_art, days = span.get(t, (0, 0))
        rows.append({"ticker": t, "name": names[t], "articles": n_art, "span_days": days,
                     "reports": len(h), "band_drop": n_band,
                     "single": 0 if h.empty else int((h["copies"] == 1).sum()),
                     "price": float(price.get(t, float("nan"))), **agg,
                     "asof": now.strftime("%Y-%m-%d")})
    res = pd.DataFrame(rows)
    store.save(res, "news_consensus")
    obs = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    if not obs.empty:
        store.save(obs.assign(pub=obs["pub"].astype(str)), "news_targets")

    # ── 요약 ────────────────────────────────────────────────────────────
    ok = res[res["n_brokers"] >= nc.MIN_BROKERS]
    print(f"\n중간값 산출 {len(ok)}/{len(res)}종목 (증권사 {nc.MIN_BROKERS}곳 이상)")
    if len(res) > 10:
        print(f"  증권사 수 분포: 0곳 {int((res['n_brokers'] == 0).sum())} · "
              f"1~2곳 {int(res['n_brokers'].between(1, 2).sum())} · "
              f"3~9곳 {int(res['n_brokers'].between(3, 9).sum())} · "
              f"10곳+ {int((res['n_brokers'] >= 10).sum())}")
        print(f"  상승여력 중앙값 {ok['upside'].median():+.1f}% · 분산 중앙값 {ok['dispersion'].median():.1f}%")
    show_rows = res if len(res) <= 12 else ok.nlargest(12, "n_brokers")
    print(f"\n{'종목':<12}{'기사':>5}{'범위':>6}{'리포트':>7}{'1회뿐':>6}{'배율탈락':>8}"
          f"{'증권사':>6}{'외국':>5}{'중간값':>12}{'현재가':>11}{'상승여력':>9}{'분산':>7}{'상향':>7}")
    for r in show_rows.itertuples():
        f = lambda v, fmt, w: (format(v, fmt) if pd.notna(v) else "—").rjust(w)  # noqa: E731
        print(f"{r.name[:10]:<12}{r.articles:>5}{str(r.span_days) + '일':>6}{r.reports:>7}{r.single:>6}"
              f"{r.band_drop:>8}{r.n_brokers:>6}{r.n_foreign:>5}{f(r.median_target, ',.0f', 12)}"
              f"{r.price:>11,.0f}{f(r.upside, '+.1f', 8)}%{f(r.dispersion, '.1f', 6)}%{f(r.revision, '+.2f', 7)}")

    for t in [x for x in args.show.split(",") if x]:
        d = obs[obs["ticker"] == t]
        if d.empty:
            print(f"\n[{names.get(t, t)}] 관측 없음")
            continue
        d = d[d["pub"] >= now - timedelta(days=nc.WINDOW_DAYS)].sort_values("pub", ascending=False)
        rep = set(nc.representatives(d, now).index)
        print(f"\n[{names[t]}] 최근 {nc.WINDOW_DAYS}일 · ◀ = 중간값에 쓰인 증권사별 대표값")
        print(f"  {'날짜':<11}{'증권사':<12}{'목표가':>12}{'이전':>11}{'방향':>5}{'복제':>5}    제목")
        for i, r in zip(d.index, d.itertuples()):
            prev = f"{r.prev_target:>11,.0f}" if pd.notna(r.prev_target) else f"{'':>11}"
            print(f"  {str(r.pub)[:10]:<11}{r.broker + ('*' if r.foreign else ''):<12}"
                  f"{r.target:>12,.0f}{prev}{(r.direction or '-'):>5}{r.copies:>5}"
                  f"{' ◀ ' if i in rep else '   '} {r.title[:36]}")
        print("  * 외국계 — 중간값에서 빼고 따로 센다 (국내 컨센서스와 기준을 맞춘다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
