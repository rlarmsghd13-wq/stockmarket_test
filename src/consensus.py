"""증권사 컨센서스 — 수동 입력분 읽기.

한경컨센서스(consensus.hankyung.com)는 목표주가·투자의견·PDF를 무료로 주지만
`robots.txt`가 전 경로 자동 수집을 금지한다. 사람이 열어보는 건 자유이므로
**매달 5~10종목만 직접 보고 CSV에 옮겨 적는다.** 그 정도는 몇 분이면 끝난다.

입력 파일: vault/_입력/컨센서스.csv  (UTF-8, 없으면 빈 결과)

    ticker,증권사,작성자,날짜,투자의견,목표주가,제목,url
    005930,현대차증권,장문수,2026-08-27,Buy,720000,CID 2026 후기,https://...

목표주가는 원 단위 숫자만. 쉼표·"원"이 섞여 있어도 파싱한다.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

COLUMNS = ["ticker", "증권사", "작성자", "날짜", "투자의견", "목표주가", "제목", "url"]
TEMPLATE = (
    "# 한경컨센서스에서 옮겨 적는 파일입니다. 자동 수집하지 않습니다.\n"
    "# https://consensus.hankyung.com/analysis/list?skinType=business\n"
    "# 종목명으로 검색 → 작성일·목표주가·투자의견·작성자·증권사를 이 표에 채우세요.\n"
    "# 목표주가는 숫자만 (쉼표·'원' 있어도 됩니다). 빈 줄과 #로 시작하는 줄은 무시됩니다.\n"
    + ",".join(COLUMNS) + "\n"
)

_NUM = re.compile(r"[^\d.\-]")
_UNSAFE = re.compile(r'[\/:*?"<>|#^\[\]]')


def _safe(name: str) -> str:
    """옵시디언 링크에 쓸 수 있게 파일명 금지문자를 정리한다 (vault.safe와 동일 규칙)."""
    return _UNSAFE.sub("-", str(name)).strip() or "무명"


def _price(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = _NUM.sub("", str(v))
    if not s or s in ("-", "."):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def path(vault_root: Path) -> Path:
    return vault_root / "_입력" / "컨센서스.csv"


def ensure(vault_root: Path) -> Path:
    """입력 파일이 없으면 헤더만 있는 템플릿을 만들어 둔다."""
    p = path(vault_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(TEMPLATE, encoding="utf-8")
    return p


def load(vault_root: Path) -> pd.DataFrame:
    p = path(vault_root)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(p, comment="#", dtype={"ticker": str},
                         encoding="utf-8", skip_blank_lines=True)
    except Exception:
        return pd.DataFrame(columns=COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df.columns = [str(c).strip() for c in df.columns]
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df["ticker"] = df["ticker"].astype(str).str.strip().str.zfill(6)
    df["목표주가"] = df["목표주가"].map(_price)
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    return df[COLUMNS].dropna(subset=["ticker"])


def summarize(df: pd.DataFrame, ticker: str, price: float | None,
              months: int = 6) -> dict | None:
    """한 종목의 컨센서스 요약. 최근 N개월 리포트만 본다."""
    if df.empty:
        return None
    d = df[df["ticker"] == str(ticker).zfill(6)].copy()
    if d.empty:
        return None
    if d["날짜"].notna().any():
        cutoff = d["날짜"].max() - pd.DateOffset(months=months)
        d = d[d["날짜"].isna() | (d["날짜"] >= cutoff)]
    tp = d["목표주가"].dropna()
    out = {
        "n": int(len(d)),
        "brokers": ", ".join(sorted(set(d["증권사"].dropna().astype(str)))),
        "target_mean": float(tp.mean()) if len(tp) else None,
        "target_min": float(tp.min()) if len(tp) else None,
        "target_max": float(tp.max()) if len(tp) else None,
        "latest": (d["날짜"].max().strftime("%Y-%m-%d")
                   if d["날짜"].notna().any() else None),
        "rows": d.sort_values("날짜", ascending=False).to_dict("records"),
    }
    if out["target_mean"] and price:
        out["upside"] = (out["target_mean"] / price - 1) * 100
        # 목표주가 편차가 크면 시장에 합의가 없다는 뜻이다.
        if out["target_max"] and out["target_min"] and out["target_mean"]:
            out["spread"] = (out["target_max"] - out["target_min"]) / out["target_mean"] * 100
    return out


def markdown(s: dict | None) -> str:
    """종목 노트에 넣을 섹션 본문."""
    if not s or not s["n"]:
        return ("_아직 입력된 리포트가 없습니다._ "
                "`vault/_입력/컨센서스.csv`에 옮겨 적으면 여기에 표시됩니다.")
    lines = []
    if s.get("target_mean"):
        head = (f"목표주가 평균 **{s['target_mean']:,.0f}원** "
                f"(최저 {s['target_min']:,.0f} · 최고 {s['target_max']:,.0f})")
        if s.get("upside") is not None:
            head += f" · 현재가 대비 **{s['upside']:+.1f}%**"
        lines.append(head)
        if s.get("spread") is not None and s["spread"] > 40:
            lines.append(f"\n> [!warning] 목표주가 편차 {s['spread']:.0f}% — "
                         f"증권사 간 합의가 없습니다. 평균값을 그대로 믿기 어렵습니다.")
    lines.append(f"\n리포트 {s['n']}건 · {s['brokers']}"
                 + (f" · 최신 {s['latest']}" if s.get("latest") else ""))
    lines.append("\n| 날짜 | 증권사 | 작성자 | 의견 | 목표주가 | 제목 |")
    lines.append("| --- | --- | --- | --- | ---: | --- |")
    for r in s["rows"][:8]:
        dt = r["날짜"].strftime("%Y-%m-%d") if pd.notna(r["날짜"]) else "—"
        tp = f"{r['목표주가']:,.0f}원" if r.get("목표주가") else "—"
        title = str(r.get("제목") or "").strip()
        if r.get("url") and isinstance(r["url"], str) and r["url"].startswith("http"):
            title = f"[{title or '리포트'}]({r['url']})"
        lines.append(f"| {dt} | {r.get('증권사') or '—'} | {r.get('작성자') or '—'} "
                     f"| {r.get('투자의견') or '—'} | {tp} | {title or '—'} |")
    return "\n".join(lines)


BASE = "https://consensus.hankyung.com/analysis/list"


def search_url(name: str, asof: str, months_back: int = 2) -> str:
    """그 종목의 한경컨센서스 검색 주소.

    자동 수집이 금지돼 있으므로 **사람이 눌러서 볼 링크**를 만들어 줄 뿐이다.
    검색 구간은 기준일에서 몇 달 전까지.
    """
    from datetime import date
    from urllib.parse import quote

    e = date.fromisoformat(asof)
    y, m = e.year, e.month - months_back
    while m <= 0:
        y, m = y - 1, m + 12
    s_ = date(y, m, 1).isoformat()
    return (f"{BASE}?skinType=business&sdate={s_}&edate={e.isoformat()}"
            f"&search_text={quote(str(name))}&pagenum=80")


def todo_block(picks: dict, asof: str, have: set[str]) -> str:
    """월간 노트에 넣을 '이번 달 조회할 리포트' 체크리스트.

    이미 CSV에 입력된 종목은 체크된 상태로 표시한다.
    """
    rows = []
    for label, rs in picks.items():
        for r in rs:
            if not r:
                continue
            done = r["ticker"] in have
            mark = "x" if done else " "
            rows.append(f"- [{mark}] [{r['name']}]({search_url(r['name'], asof)}) "
                        f"`{r['ticker']}` · {label}"
                        + ("" if not done else " — 입력됨"))
    if not rows:
        return "_대상 종목이 없습니다._"
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# 지표 vs 컨센서스 대조
#
# 내 점수와 증권사 목표주가는 **서로 다른 방법으로 만든 독립적인 신호**다.
# 같은 결론이면 신뢰도가 올라가고, 갈리면 둘 중 하나가 무언가를 놓치고 있다.
# 갈리는 종목이 확인할 가치가 가장 큰 종목이다.
#
# 컨센서스를 점수에 섞지는 않는다 — 예측력이 검증되지 않았고,
# 검증 안 된 신호를 검증된 가중치에 섞으면 희석된다.
# ---------------------------------------------------------------------------
GAP_RANK = 2          # 순위가 이만큼 벌어지면 괴리로 본다
GAP_UPSIDE_HIGH = 30  # 상승여력 30% 이상이면 "컨센 강세"
GAP_UPSIDE_LOW = 5    # 5% 미만이면 "컨센 약세"


def compare(picks: list[dict], df: pd.DataFrame,
            summaries: dict[str, dict] | None = None) -> list[dict]:
    """전략별 상위 종목에 대해 내 점수 순위 vs 컨센서스 상승여력 순위.

    summaries: {ticker: news_summary()} — 있으면 수동 CSV보다 우선한다.
    """
    rows = []
    for r in picks:
        if not r:
            continue
        s = (summaries or {}).get(r["ticker"]) or summarize(df, r["ticker"], r.get("price"))
        rows.append({
            "ticker": r["ticker"], "name": r["name"], "score": r["score"],
            "verdict": r["verdict"]["label"],
            "upside": (s or {}).get("upside"),
            "n": (s or {}).get("n", 0),
            "spread": (s or {}).get("spread"),
        })
    have = [x for x in rows if x["upside"] is not None]
    if len(have) >= 2:
        by_score = sorted(have, key=lambda x: -x["score"])
        by_up = sorted(have, key=lambda x: -x["upside"])
        sr = {x["ticker"]: i + 1 for i, x in enumerate(by_score)}
        ur = {x["ticker"]: i + 1 for i, x in enumerate(by_up)}
        for x in rows:
            x["rank_score"] = sr.get(x["ticker"])
            x["rank_upside"] = ur.get(x["ticker"])
            if x["rank_score"] and x["rank_upside"]:
                d = x["rank_upside"] - x["rank_score"]
                x["gap"] = d
                if abs(d) < GAP_RANK:
                    x["note"] = "일치"
                elif d > 0:
                    x["note"] = "내 점수 높음 · 컨센 낮음"
                else:
                    x["note"] = "컨센 높음 · 내 점수 낮음"
            else:
                x["gap"], x["note"] = None, None
    return rows


def compare_markdown(rows: list[dict]) -> str:
    have = [x for x in rows if x.get("upside") is not None]
    if not have:
        return ("_컨센서스 입력이 없어 대조할 수 없습니다._ "
                "`_입력/컨센서스.csv`를 채우면 여기에 표시됩니다.")
    lines = [
        "내 점수와 증권사 목표주가는 서로 다른 방법으로 만든 **독립적인 신호**입니다. "
        "같은 결론이면 신뢰도가 올라가고, 갈리면 둘 중 하나가 무언가를 놓치고 있습니다.",
        "",
        "| 종목 | 판정 | 내 점수 | 순위 | 컨센 상승여력 | 순위 | 대조 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for x in sorted(rows, key=lambda y: -y["score"]):
        up = f"{x['upside']:+.1f}%" if x.get("upside") is not None else "—"
        rs = str(x.get("rank_score") or "—")
        ru = str(x.get("rank_upside") or "—")
        note = x.get("note") or ("리포트 없음" if not x.get("n") else "—")
        mark = "⚠ " if note and "낮음" in note else ""
        lines.append(f"| [[{_safe(x['name'])}]] | {x['verdict']} | {x['score']:.1f} | {rs} "
                     f"| {up} | {ru} | {mark}{note} |")
    gaps = [x for x in have if x.get("note") and "낮음" in x["note"]]
    if gaps:
        lines.append("")
        lines.append("> [!warning] 확인할 종목")
        for x in gaps:
            lines.append(f"> - **{x['name']}** — {x['note']}. "
                         f"리포트를 읽고 내 지표가 놓친 게 있는지 보세요.")
    lines.append("")
    lines.append("> [!note] 컨센서스는 점수에 넣지 않습니다")
    lines.append("> 목표주가의 예측력이 아직 검증되지 않았습니다. "
                 "검증 안 된 신호를 검증된 가중치에 섞으면 희석됩니다 — "
                 "모멘텀 부문(IC −0.003)과 3년 CAGR 게이트(IC −0.029)에서 이미 겪은 일입니다.")
    lines.append("> 매달 쌓이면 12개월 뒤 예측력을 검증할 수 있습니다.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 뉴스 기반 컨센서스 (scripts/21_news_consensus.py 산출물)
#
# 수동 CSV는 사람이 매달 옮겨 적어야 했다. 뉴스 검색 API로 인용된 목표가를
# 모으면 유니버스 199종목 중 약 150종목이 증권사 3곳 이상으로 채워진다
# (네이버 증권 컨센서스 대조: KB금융 −2.4%, 현대차 +2.6%).
# 수동 CSV는 뉴스로 안 잡히는 종목을 보충하는 용도로 남긴다.
# ---------------------------------------------------------------------------
def _kst(ts) -> str:
    """UTC로 저장된 게시 시각을 한국 날짜로 — 새벽 기사가 전날로 찍히지 않게."""
    t = pd.Timestamp(ts)
    return (t.tz_convert("Asia/Seoul") if t.tzinfo else t).strftime("%Y-%m-%d")


def news_summary(obs: pd.DataFrame, cons: pd.DataFrame, ticker: str,
                 price: float | None, asof) -> dict | None:
    """한 종목의 뉴스 컨센서스 — 증권사별 대표값 기준.

    obs  : store 'news_targets' (복제 합치고 배율 걸러낸 관측치)
    cons : store 'news_consensus' (종목별 집계, 상승여력 백분위 계산용)
    """
    from . import news_consensus as nc

    if obs is None or obs.empty:
        return None
    d = obs[obs["ticker"] == str(ticker).zfill(6)]
    if d.empty:
        return None
    rep = nc.representatives(d, asof)
    if rep.empty:
        return None
    agg = nc.aggregate(d, price, asof)
    out = {
        "source": "뉴스",
        "n": int(agg["n_brokers"]),
        "n_foreign": int(agg["n_foreign"]),
        "brokers": ", ".join(sorted(rep["broker"])),
        "target_median": agg["median_target"],
        "target_min": float(rep["target"].min()),
        "target_max": float(rep["target"].max()),
        "upside": agg["upside"],
        "dispersion": agg["dispersion"],
        "revision": agg["revision"],
        "ups": int(rep["direction"].eq("상향").sum()),
        "downs": int(rep["direction"].eq("하향").sum()),
        "latest": _kst(rep["pub"].max()),
        "rows": rep.sort_values("pub", ascending=False).to_dict("records"),
    }
    # 목표가는 체계적으로 낙관적이라 절대값보다 **유니버스 안 순위**가 의미 있다
    if out["upside"] is not None and cons is not None and not cons.empty:
        pool = cons[cons["n_brokers"] >= nc.MIN_BROKERS]["upside"].dropna()
        if len(pool) >= 10:
            out["upside_pct"] = float((pool < out["upside"]).mean() * 100)
            out["upside_median_all"] = float(pool.median())
    # compare()가 쓰는 키 이름을 맞춘다
    out["target_mean"] = out["target_median"]
    out["spread"] = out["dispersion"]
    return out


def news_markdown(s: dict | None) -> str:
    """종목 노트의 '증권사 시각' 섹션 본문."""
    if not s or not s.get("n"):
        return ""
    lines = []
    if s.get("target_median"):
        head = f"증권사 **{s['n']}곳** · 목표가 중간값 **{s['target_median']:,.0f}원**"
        if s.get("upside") is not None:
            head += f" · 현재가 대비 {s['upside']:+.1f}%"
            if s.get("upside_pct") is not None:
                pct = s["upside_pct"]
                head += (f" (유니버스 상위 {100 - pct:.0f}%)" if pct >= 50
                         else f" (유니버스 하위 {pct:.0f}%)")
        lines.append(head)
        tail = []
        if s.get("dispersion") is not None:
            tone = "의견 일치" if s["dispersion"] < 10 else ("엇갈림" if s["dispersion"] > 25 else "보통")
            tail.append(f"의견 분산 {s['dispersion']:.0f}% ({tone})")
        tail.append(f"최근 90일 상향 {s['ups']} · 하향 {s['downs']}")
        if s.get("n_foreign"):
            tail.append(f"외국계 {s['n_foreign']}곳 별도")
        lines.append(" · ".join(tail))
    else:
        lines.append(f"증권사 {s['n']}곳뿐이라 중간값을 내지 않았습니다 (3곳 이상 필요).")
    lines += ["", "| 날짜 | 증권사 | 목표가 | 변화 | 출처 |", "| --- | --- | ---: | --- | --- |"]
    for r in s["rows"][:12]:
        chg = r.get("direction") or "—"
        if r.get("prev_target") == r.get("prev_target") and r.get("prev_target"):
            chg += f" (← {r['prev_target']:,.0f})"
        link = f"[기사]({r['link']})" if r.get("link") else ""
        lines.append(f"| {_kst(r['pub'])} | {r['broker']} | {r['target']:,.0f} | {chg} | {link} |")
    med_all = s.get("upside_median_all")
    lines += ["", "> [!note] 읽는 법",
              "> 증권사 목표가는 체계적으로 낙관적입니다"
              + (f" — 유니버스 전체 상승여력 중앙값이 {med_all:+.0f}%입니다." if med_all is not None else ".")
              + " 절대값보다 **다른 종목 대비 순위**와 **상향·하향 방향**을 보세요.",
              "> 주가가 급락한 뒤엔 목표가가 늦게 따라와서 상승여력이 부풀려 보입니다."]
    return "\n".join(lines)
