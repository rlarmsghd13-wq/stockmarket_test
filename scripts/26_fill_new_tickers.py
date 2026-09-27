"""유니버스 합집합의 재무·시세·트랙 구멍 메우기.

    python scripts/26_fill_new_tickers.py --what report   # 무엇이 빠졌는지만
    python scripts/26_fill_new_tickers.py --what dart     # 재무 (DART, 일 2만 회)
    python scripts/26_fill_new_tickers.py --what krx      # 시세 (KRX, 일 800회)
    python scripts/26_fill_new_tickers.py --what tracks   # 트랙 판정 (DART)

**한 번에 하나만 돌린다.** KRX 차단의 직접 원인이 두 작업 동시 실행이었다.

왜 필요한가
    25_fill_universe.py 로 스냅샷 구멍을 메우면 후보 종목이 늘어난다. 그런데
    종목만 늘려도 소용이 없다 — 재무가 없으면 패널에 못 들어가고, 시세가 없으면
    수익률을 못 잰다.

    같이 드러난 더 큰 구멍: `fin_ttm_hist`는 **2017~2022년만**, `fin_ttm`은
    **현재 상위 200종목만** 담고 있다. 그래서 2023년 이후 패널에 들어가는
    나머지 200여 종목이 **2022년 재무로 채점되고 있었다.** 미래 정보는 아니지만
    3년 묵은 숫자로 성장성을 재는 셈이라 점수가 의미를 잃는다.
    이 스크립트의 `--what dart` 는 합집합 전 종목 × 전 연도를 다시 만든다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import config                                                      # noqa: E402
from src import dart, krx, quarterly, store, tracks, ttm, universe  # noqa: E402

YEAR_START = 2017          # TTM은 전년 연간이 있어야 계산된다
PRICE_START = "2017-01-01"
PRICE_SPLIT = "2021-12-01"  # 10년을 한 번에 요청하면 KRX가 조용히 빈 값을 준다
OVERLAP_START = "2023-06-01"
EMPTY_YEARS_STOP = 2        # 데이터가 시작된 뒤 연속 빈 연도 → 상장폐지로 보고 중단
STALE_PRICE_DAYS = 60       # 이보다 오래 끊기면 합병·상장폐지를 의심한다
SAVE_EVERY = 25


def log(m: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


# ---------------------------------------------------------------------------
# 현황
# ---------------------------------------------------------------------------
def status() -> pd.DataFrame:
    """합집합 종목별로 재무 최신 연도 · 시세 종료일 · 트랙 유무."""
    snaps = universe.load_snapshots()
    inc = snaps[snaps["included"]]
    names = snaps.sort_values("asof_date").groupby("ticker")["name"].last()
    first = inc.groupby("ticker")["asof_date"].min()
    last = inc.groupby("ticker")["asof_date"].max()

    fin = []
    for n in ("fin_ttm_hist", "fin_ttm"):
        if store.exists(n):
            fin.append(store.load(n)[["ticker", "year"]])
    fin_year = (pd.concat(fin).groupby("ticker")["year"].max()
                if fin else pd.Series(dtype=float))

    px = []
    for n in ("prices_daily_hist", "prices_daily"):
        if store.exists(n):
            d = store.load(n)[["ticker", "date"]]
            d["date"] = pd.to_datetime(d["date"])
            px.append(d)
    px_last = (pd.concat(px).groupby("ticker")["date"].max()
               if px else pd.Series(dtype="datetime64[ns]"))

    tk = set(store.load("tracks")["ticker"]) if store.exists("tracks") else set()

    rows = []
    for t in sorted(set(inc["ticker"])):
        rows.append({
            "ticker": t, "name": names.get(t, ""),
            "first_in": first.get(t, ""), "last_in": last.get(t, ""),
            "fin_year": fin_year.get(t, float("nan")),
            "px_last": px_last.get(t, pd.NaT),
            "has_track": t in tk,
        })
    return pd.DataFrame(rows)


def report() -> int:
    s = status()
    today = date.today()
    print(f"유니버스 합집합 {len(s)}종목\n")

    print("① 재무 최신 연도 분포 — 그 종목이 마지막으로 편입된 해와 비교한다")
    print(f"  {'재무 최신':<10}{'종목수':>7}")
    for y, n in s["fin_year"].value_counts(dropna=False).sort_index().items():
        print(f"  {str(int(y)) if pd.notna(y) else '없음':<10}{n:>7}")

    # 마지막 편입 시점보다 재무가 뒤처진 종목 = 묵은 숫자로 채점되는 종목
    s["last_in_year"] = pd.to_numeric(s["last_in"].str[:4], errors="coerce")
    stale = s[s["fin_year"].isna() | (s["fin_year"] < s["last_in_year"])]
    print(f"\n② 편입 시점보다 재무가 오래된 종목 {len(stale)}개 — 묵은 숫자로 채점된다")
    for r in stale.head(20).itertuples():
        fy = "없음" if pd.isna(r.fin_year) else f"{int(r.fin_year)}년"
        print(f"  {r.ticker} {str(r.name):<18} 재무 {fy:<7} 마지막 편입 {r.last_in}")
    if len(stale) > 20:
        print(f"  ... 외 {len(stale)-20}개")

    # 기준일을 빡빡하게 잡으면 전부 "낡음"으로 나온다. 마지막 수집일과 오늘
    # 사이가 며칠 벌어지는 건 정상이고, 합병·폐지로 정말 끊긴 것과 구분해야 한다.
    cut = pd.Timestamp(today) - pd.Timedelta(days=STALE_PRICE_DAYS)
    no_px = s[s["px_last"].isna()]
    old_px = s[s["px_last"].notna() & (s["px_last"] < cut)]
    print(f"\n③ 시세 — 아예 없음 {len(no_px)}개 · "
          f"{STALE_PRICE_DAYS}일 넘게 끊김 {len(old_px)}개(합병·폐지 포함)")
    if len(no_px):
        print("  없음: " + ", ".join(f"{r.ticker}({r.name})" for r in no_px.head(15).itertuples()))

    print(f"\n④ 트랙 미판정 {int((~s['has_track']).sum())}개 "
          "— 판정이 없으면 T1(일반)으로 취급되어 은행·지주의 지표가 왜곡된다")
    return 0


# ---------------------------------------------------------------------------
# DART 재무
# ---------------------------------------------------------------------------
def fetch_dart(limit: int) -> int:
    years = list(range(YEAR_START, date.today().year + 1))
    tickers = universe.included_tickers()
    cc = dart.corp_codes()
    cmap = dict(cc[["ticker", "corp_code"]].values)
    todo = [t for t in tickers if cmap.get(t)]
    log(f"대상 {len(todo)}종목 × {years[0]}~{years[-1]} ({len(years)}년)")
    log(f"이미 받은 원본은 디스크 캐시를 타므로 네트워크 호출은 그보다 훨씬 적다")

    q_all, t_all, fails, done = [], [], [], 0
    t0 = time.monotonic()

    # 진행 중에는 **임시 이름에만** 쓴다. 중간 저장이 본 파일을 덮으면 중단된
    # 실행이 404종목짜리 재무를 3종목짜리로 바꿔놓는다. 다 끝났고 표본이
    # 줄지 않았을 때만 승격한다.
    def flush():
        if not q_all:
            return
        store.save(pd.concat(q_all, ignore_index=True), "fin_quarterly_hist_partial")
        store.save(pd.concat(t_all, ignore_index=True), "fin_ttm_hist_partial")

    for i, tk in enumerate(todo, 1):
        if limit and done >= limit:
            log(f"--limit {limit} 도달 — 중단합니다. 다음 실행이 이어받습니다.")
            break
        parts, started, empty_run = [], False, 0
        try:
            for y in years:
                q = quarterly.fetch_company(cmap[tk], tk, [y])
                if q.empty:
                    if started:
                        empty_run += 1
                        # 상장폐지된 종목의 남은 연도를 계속 두드릴 이유가 없다
                        if empty_run >= EMPTY_YEARS_STOP:
                            break
                    continue
                started, empty_run = True, 0
                parts.append(q)
            if parts:
                q = pd.concat(parts, ignore_index=True).sort_values("period")
                q_all.append(q)
                t_all.append(ttm.build(q))
            done += 1
        except Exception as exc:
            fails.append((tk, f"{type(exc).__name__}: {str(exc)[:60]}"))
        if i % SAVE_EVERY == 0:
            flush()
            el = time.monotonic() - t0
            log(f"  {i}/{len(todo)}  경과 {el/60:.1f}분  "
                f"남은 예상 {el/i*(len(todo)-i)/60:.1f}분")

    flush()
    if not t_all:
        log("받은 재무가 없습니다.")
        return done
    t = pd.concat(t_all, ignore_index=True)
    log(f"임시 저장: fin_ttm_hist_partial {len(t):,}행 / {t.ticker.nunique()}종목 "
        f"/ {t['year'].min()}~{t['year'].max()}년")
    if fails:
        log(f"실패 {len(fails)}종목: " + ", ".join(f"{x}({e})" for x, e in fails[:8]))

    incomplete = done < len(todo)
    old_n = (store.load("fin_ttm_hist")["ticker"].nunique()
             if store.exists("fin_ttm_hist") else 0)
    new_n = t["ticker"].nunique()
    if incomplete:
        log(f"전체 {len(todo)}종목 중 {done}종목만 처리했습니다 — 승격하지 않습니다.")
        log("  같은 명령을 다시 실행하세요. 받아둔 원본은 캐시를 타므로 빠릅니다.")
        return done
    if new_n < old_n:
        log(f"[중단] 종목 수가 {old_n} → {new_n}으로 줄었습니다. 승격하지 않습니다.")
        log("  fin_ttm_hist_partial 을 직접 확인하세요.")
        return done
    store.save(pd.concat(q_all, ignore_index=True), "fin_quarterly_hist")
    store.save(t, "fin_ttm_hist")
    log(f"승격: fin_ttm_hist {old_n} → {new_n}종목")
    return done


# ---------------------------------------------------------------------------
# KRX 시세
# ---------------------------------------------------------------------------
def fetch_krx(limit: int, end: str) -> int:
    s = status()
    cut = pd.Timestamp(end) - pd.Timedelta(days=STALE_PRICE_DAYS)
    fresh = [(r.ticker, "new") for r in s[s["px_last"].isna()].itertuples()]
    extend = [(r.ticker, "extend") for r in
              s[s["px_last"].notna() & (s["px_last"] < cut)].itertuples()]
    todo = fresh + extend
    log(f"시세 대상 {len(todo)}종목 (신규 {len(fresh)} · 연장 {len(extend)})")
    log(f"예상 KRX 호출 {len(fresh)*2 + len(extend)}회 "
        f"(소프트 한도 {krx.DAILY_SOFT_LIMIT}/일)")
    if not todo:
        return 0

    ok, msg = krx.available()
    log(f"KRX 상태: {msg}")
    if not ok:
        log("차단 중입니다 — 아무것도 수집하지 않습니다.")
        return 0

    got, empty, fails, done = [], [], [], 0
    t0 = time.monotonic()
    for i, (tk, kind) in enumerate(todo, 1):
        if limit and done >= limit:
            log(f"--limit {limit} 도달 — 중단합니다.")
            break
        if krx.call_count() + 2 > krx.DAILY_SOFT_LIMIT:
            log(f"소프트 한도 근접({krx.call_count()}회) — 중단합니다. 내일 이어서.")
            break
        windows = ([(PRICE_START, PRICE_SPLIT), (PRICE_SPLIT, end)]
                   if kind == "new" else [(OVERLAP_START, end)])
        try:
            parts = [krx.ohlcv(tk, a, b) for a, b in windows]
            parts = [p for p in parts if not p.empty]
            if parts:
                got.append(pd.concat(parts, ignore_index=True))
            else:
                empty.append(tk)
            done += 1
        except Exception as exc:
            fails.append((tk, str(exc)[:60]))
        if i % 25 == 0:
            el = time.monotonic() - t0
            log(f"  {i}/{len(todo)}  경과 {el/60:.1f}분  KRX {krx.call_count()}회")

    if not got:
        log("받은 시세가 없습니다.")
        return done

    hist = store.load("prices_daily_hist")
    hist["date"] = pd.to_datetime(hist["date"])
    new = pd.concat(got, ignore_index=True)
    new["date"] = pd.to_datetime(new["date"])
    keep = [c for c in hist.columns if c in new.columns]
    merged = (pd.concat([hist, new[keep]], ignore_index=True)
                .drop_duplicates(["ticker", "date"], keep="last")
                .sort_values(["ticker", "date"]))
    store.save(merged, "prices_daily_hist")
    log(f"저장: prices_daily_hist {len(merged):,}행 / {merged.ticker.nunique()}종목")
    log(f"  성공 {new.ticker.nunique()}종목 · 빈 응답 {len(empty)} · 실패 {len(fails)}")
    if empty:
        log("  빈 응답(상장폐지·거래정지 후보): " + ", ".join(empty[:20]))
    if fails:
        log("  실패: " + ", ".join(t for t, _ in fails[:10]))
    return done


# ---------------------------------------------------------------------------
# 트랙 판정
# ---------------------------------------------------------------------------
def fetch_tracks(limit: int) -> int:
    s = status()
    todo = s[~s["has_track"]]
    log(f"트랙 미판정 {len(todo)}종목")
    if todo.empty:
        return 0

    fin = pd.concat([store.load(n) for n in ("fin_quarterly_hist", "fin_quarterly")
                     if store.exists(n)], ignore_index=True)
    cc = dart.corp_codes()
    cmap = dict(cc[["ticker", "corp_code"]].values)

    rows, fails, done = [], [], 0
    for i, r in enumerate(todo.itertuples(), 1):
        if limit and done >= limit:
            log(f"--limit {limit} 도달 — 중단합니다.")
            break
        code = cmap.get(r.ticker)
        if not code:
            fails.append((r.ticker, "corp_code 없음"))
            continue
        # 상장폐지 종목은 2025년 재무가 없다. 그 종목이 가진 마지막 연간을 쓴다.
        g = fin[(fin["ticker"] == r.ticker) & (fin["quarter"] == 4)]
        fy = int(g["year"].max()) if not g.empty else None
        try:
            annual_row = (g[g["year"] == fy].iloc[0].to_dict()
                          if fy is not None else None)
            raw_bs = dart.financials(code, fy, config.REPRT_ANNUAL, "CFS") \
                if fy is not None else pd.DataFrame()
            if not raw_bs.empty:
                raw_bs = raw_bs[raw_bs["sj_div"] == "BS"]
            rows.append({**tracks.assign(code, r.ticker, str(r.name),
                                         annual_row=annual_row, raw_bs=raw_bs),
                         "name": r.name, "market_cap": float("nan")})
            done += 1
        except Exception as exc:
            fails.append((r.ticker, f"{type(exc).__name__}: {str(exc)[:50]}"))
        if i % 25 == 0:
            log(f"  {i}/{len(todo)}")

    if not rows:
        log("판정 결과가 없습니다.")
        return 0
    new = pd.DataFrame(rows)
    old = store.load("tracks") if store.exists("tracks") else pd.DataFrame()
    merged = (pd.concat([old, new], ignore_index=True)
                .drop_duplicates(["ticker"], keep="first"))   # 기존 판정을 지키다
    store.save(merged, "tracks")
    log(f"저장: tracks {len(merged)}종목 (신규 {len(new)})")
    print("\n신규 판정 트랙 분포:")
    for tr, n in new["track"].value_counts().items():
        print(f"  {tr}  {n:>3}종목")
    if fails:
        log(f"실패 {len(fails)}종목: " + ", ".join(f"{t}({e})" for t, e in fails[:8]))
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["report", "dart", "krx", "tracks"],
                    required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--end", default=date.today().isoformat())
    args = ap.parse_args()

    if args.what == "report":
        return report()
    if args.what == "dart":
        n = fetch_dart(args.limit)
    elif args.what == "krx":
        n = fetch_krx(args.limit, args.end)
    else:
        n = fetch_tracks(args.limit)
    log(f"완료 — {n}종목 처리")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
