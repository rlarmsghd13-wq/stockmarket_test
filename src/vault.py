"""옵시디언 볼트 생성 — 월 1회 실행.

구조
    vault/
      README.md
      월간리포트/2026-09.md        업종 상위 5 + 전략별 상위 5
      가치투자/ 성장주/ 급등주/     종목당 노트 1개
      _이력/전략변동.md            폴더 이동 로그

**종목 노트는 하나만 유지하고 월별 이력을 안에 누적한다.** 전략이 바뀌면
노트를 옮기고 변동 이력에 한 줄을 남긴다. 그래야 "이 종목이 언제 가치에서
성장으로 넘어갔는지"가 한 파일에서 보이고, 옵시디언 링크도 끊기지 않는다.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# 급등주는 검증에서 종합점수 IC +0.012로 예측력이 없어 제거했다.
FOLDER = {"value": "가치투자", "growth": "성장주"}
CATS = [("growth", "성장성"), ("profitability", "수익성"), ("stability", "안정성"),
        ("valuation", "밸류에이션"), ("momentum", "모멘텀·수급")]

_FM = re.compile(r"^---\n(.*?)\n---\n", re.S)
_SECTION = re.compile(r"\n## (.+?)\n(.*?)(?=\n## |\Z)", re.S)


def safe(name: str) -> str:
    """옵시디언 파일명에서 못 쓰는 문자를 정리한다."""
    return re.sub(r'[\\/:*?"<>|#^\[\]]', "-", str(name)).strip() or "무명"


def read_note(path: Path) -> tuple[dict, dict[str, str]]:
    """기존 노트에서 frontmatter와 섹션별 본문을 뽑는다."""
    if not path.exists():
        return {}, {}
    text = path.read_text(encoding="utf-8")
    fm: dict[str, str] = {}
    m = _FM.match(text)
    body = text
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fm[k.strip()] = v.strip().strip('"')
        body = text[m.end():]
    sections = {t.strip(): b.strip() for t, b in _SECTION.findall("\n" + body)}
    return fm, sections


def parse_rows(md_table: str) -> list[str]:
    """마크다운 표에서 데이터 행만. 헤더·구분선은 버린다."""
    out = []
    for line in md_table.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        if set(s.replace("|", "").replace(" ", "")) <= set("-:"):
            continue
        if s.startswith("| 월 ") or s.startswith("|월"):
            continue
        out.append(s)
    return out


def find_existing(vault: Path, ticker: str) -> Path | None:
    """세 전략 폴더 어디에 있든 이 종목 노트를 찾는다."""
    for f in FOLDER.values():
        d = vault / f
        if not d.exists():
            continue
        for p in d.glob("*.md"):
            fm, _ = read_note(p)
            if str(fm.get("종목코드", "")).strip('"') == ticker:
                return p
    return None


def _fmt(v, f="{:,.0f}", dash="—"):
    return dash if v is None or (isinstance(v, float) and pd.isna(v)) else f.format(v)


def stock_note(r: dict, sector: str, month: str,
               prev_rows: list[str], prev_moves: list[str],
               moved_from: str | None, consensus_md: str = "") -> str:
    """종목 노트 전체 텍스트. 월별 이력은 누적한다."""
    cat = {c["key"]: c for c in r["categories"]}
    strat = FOLDER[r["strategy"]]

    def cs(k):
        c = cat.get(k)
        return "—" if not c or c["score"] is None else f"{c['score']:.0f}"

    def cg(k):
        c = cat.get(k)
        return "" if not c or not c["grade"] else c["grade"]

    row = (f"| {month} | {strat} | **{r['verdict']['label']}** | {r['score']:.1f} | " +
           " | ".join(cs(k) for k, _ in CATS) + " | " +
           (f"{r['band']['per_pos']:.0f}p" if r["band"]["per_pos"] is not None else "—") +
           f" | {_fmt(r['price'])}원 |")
    rows = [x for x in prev_rows if not x.startswith(f"| {month} |")] + [row]
    rows = sorted(rows, reverse=True)

    moves = list(prev_moves)
    if moved_from:
        moves.append(f"| {month} | {moved_from} → **{strat}** | {r['score']:.1f} | "
                     f"전략 귀속 변경 |")
    elif not prev_rows:
        moves.append(f"| {month} | — → **{strat}** | {r['score']:.1f} | 신규 편입 |")

    grade_tbl = "\n".join(
        f"| {kr} | {cs(k)} | {cg(k)} |" for k, kr in CATS)
    contrib = "\n".join(
        f"| {c['cat']} | {c['score']:.0f} | {c['weight']}% | {c['points']:.1f} |"
        for c in r["contrib"] if c["points"] is not None)
    exits = "\n".join(
        f"| {e['case']} | {e['trigger']} | {e['action']} | {e['basis']} |"
        for e in r["exit_plan"])
    b = r["band"]
    band_line = ("밴드 산출 불가" if b["per_pos"] is None else
                 f"2년 PER 밴드 **{b['per_pos']:.0f}백분위** — "
                 f"25p {b['per_p25']:.1f}배 · 50p {b['per_p50']:.1f}배 · 75p {b['per_p75']:.1f}배")

    flags = []
    if not r["flags"]["ttm_reliable"]:
        flags.append("구조변경 직후 — TTM 신뢰 불가")
    if r["flags"]["loss_streak"]:
        flags.append(f"영업적자 {r['flags']['loss_streak']}년 연속")
    if r["flags"]["stability_distorted"]:
        flags.append("금융 부문 연결 — 안정성 지표 왜곡")
    if r["coverage"] is not None and r["coverage"] < 4:
        flags.append(f"부문 커버리지 {r['coverage']}/4")
    flag_block = ("\n> [!warning] 주의\n" + "\n".join(f"> - {x}" for x in flags) + "\n"
                  if flags else "")

    return f"""---
종목명: {r['name']}
종목코드: "{r['ticker']}"
시장: {r['market']}
트랙: {r['track']}
업종: {sector}
현재전략: {strat}
최종갱신: {month}
최종점수: {r['score']:.1f}
판정: {r['verdict']['label']}
tags: [종목, {strat}, {r['track']}]
---

# {r['name']} ({r['ticker']})

> [!info] {month} 기준
> **{r['verdict']['label']}** · {r['score']:.1f}점 · {strat} · {_fmt(r['price'])}원 · 시총 {r['market_cap']/1e12:.1f}조
> 재무 {r['fin_period']} (접수 {r['rcept_dt']}) · 업종 {sector}
{flag_block}
## 부문 점수

| 부문 | 점수 | 등급 |
| --- | ---: | :---: |
{grade_tbl}

{band_line}

## 점수 기여도

| 부문 | 점수 | 가중치 | 기여 |
| --- | ---: | ---: | ---: |
{contrib}

## 판정

> [!{('tip' if r['verdict']['code'] == 'buy' else 'note')}] {r['verdict']['label']}
> {r['verdict']['why']}
> 매수는 **점수만으로** 정합니다. 가격 타이밍을 잡지 않습니다.

## 매도 계획

| 구분 | 트리거 | 대응 | 근거 |
| --- | --- | --- | --- |
{exits}

## 증권사 시각

{consensus_md or "_뉴스에 인용된 목표가가 없고, 직접 입력한 리포트도 없습니다._"}

## 찬성 근거

{chr(10).join('- ' + x for x in r['pros'])}

## 반대 근거

{chr(10).join('- ' + x for x in r['cons'])}

## 논거 훼손 조건

발생 시 가격과 무관하게 청산을 검토한다.

{chr(10).join('- ' + x for x in r['breakers'])}

## 전략 변동 이력

| 월 | 변동 | 점수 | 비고 |
| --- | --- | ---: | --- |
{chr(10).join(moves)}

## 월별 이력

| 월 | 전략 | 판정 | 점수 | 성장 | 수익 | 안정 | 밸류 | 모멘텀 | 밴드 | 주가 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

---
*자동 생성 · `screener/scripts/08_build_vault.py` · 투자 권유가 아닙니다.*
"""


def month_note(month: str, asof: str, sectors: pd.DataFrame,
               picks: dict[str, list[dict]], moves: list[str],
               universe_n: int, gate_counts: dict, todo: str = "",
               compare_md: str = "") -> str:
    sec_tbl = "\n".join(
        f"| {i} | {r.sector_name} | {r.n}종목 | {r.mean_score:.1f} | "
        f"{r.top_name} {r.top_score:.1f} |"
        for i, r in enumerate(sectors.itertuples(), 1))

    blocks = []
    for s, label in FOLDER.items():
        rs = picks.get(s, [])
        if not rs:
            blocks.append(f"### {label}\n\n게이트 통과 종목 없음\n")
            continue
        lines = "\n".join(
            f"| {i} | [[{safe(r['name'])}]] | **{r['verdict']['label']}** | {r['score']:.1f} | "
            + " | ".join(
                ("—" if c["score"] is None else f"{c['score']:.0f}")
                for c in r["categories"])
            + " | "
            + (f"{r['band']['per_pos']:.0f}p" if r["band"]["per_pos"] is not None else "—")
            + f" | {_fmt(r['price'])}원 |"
            for i, r in enumerate(rs, 1))
        blocks.append(
            f"### {label}  <small>게이트 통과 {gate_counts.get(s, 0)}종목</small>\n\n"
            "| # | 종목 | 판정 | 점수 | 성장 | 수익 | 안정 | 밸류 | 모멘텀 | 밴드 | 주가 |\n"
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
            f"{lines}\n")

    move_block = ("\n".join(moves) if moves else "| — | 변동 없음 | | |")

    return f"""---
월: {month}
기준일: {asof}
유니버스: {universe_n}
tags: [월간리포트]
---

# {month} 월간 스크리닝

> [!info] 기준
> {asof} 기준 · KOSPI·KOSDAQ 시총 상위 {universe_n}종목
> 재무는 각 종목의 **공시 접수일 기준**으로 그 시점에 알 수 있었던 값만 사용

## 상위 업종 5

전략 무관 종합점수(각 종목의 최고 전략 점수) 평균. 3종목 이상인 업종만.

| # | 업종 | 종목 수 | 평균 점수 | 최고 종목 |
| ---: | --- | ---: | ---: | --- |
{sec_tbl}

## 전략별 상위 5

{chr(10).join(blocks)}

## 지표 vs 증권사 컨센서스

{compare_md}

## 리포트 조회 (수동 — 뉴스로 안 잡힌 종목만)

증권사 목표가는 뉴스 검색 API로 자동 수집합니다(증권사 3곳 이상일 때만 중간값 산출).
아래는 **뉴스에 인용이 부족한 종목**뿐입니다. 한경컨센서스는 자동 수집이 금지돼 있어
링크를 눌러 `_입력/컨센서스.csv`에 옮겨 적으면 다음 실행 때 반영됩니다.

{todo}

## 이번 달 변동

| 월 | 변동 | 점수 | 비고 |
| --- | --- | ---: | --- |
{move_block}

---

> [!warning] 읽는 법
> 점수는 **동종업종 내 백분위**를 가중평균한 값입니다. 8년 백테스트에서 가치투자는
> 벤치마크에 졌고 성장주는 이겼지만 낙폭이 두 배였습니다([[백테스트_2026-09]]).
> **정렬 기준이지 예측값이 아닙니다.** 규칙 비교는 [[성과추적]]에서 매달 쌓고 있습니다.
> 증권사 목표가는 **참고용**이며 점수에 들어가지 않습니다. 공시·IR 정성 근거는 아직 없습니다.

*자동 생성 · 투자 권유가 아닙니다. 최종 판단과 책임은 본인에게 있습니다.*
"""


HELD_LABELS = ("매수", "보유", "비중 축소")


def was_held(prev_rows: list[str]) -> bool:
    """직전 월 판정이 매수·보유·비중축소였으면 보유 중으로 본다.

    히스테리시스를 적용하려면 "지금 들고 있는가"를 알아야 하는데,
    별도 포트폴리오 파일 없이 노트의 월별 이력에서 읽어온다.
    """
    if not prev_rows:
        return False
    latest = sorted(prev_rows, reverse=True)[0]
    return any(f"**{lab}**" in latest for lab in HELD_LABELS)


def defer_months(prev_rows: list[str]) -> int:
    """최근 연속으로 '매도 보류' 판정이 몇 개월 이어졌는가.

    시장 요인이라며 무한정 들고 있으면 진짜 부실을 놓친다.
    regime.MAX_DEFER_MONTHS를 넘으면 보류가 풀린다.
    """
    n = 0
    for line in sorted(prev_rows, reverse=True):
        if "매도 보류" in line:
            n += 1
        else:
            break
    return n
