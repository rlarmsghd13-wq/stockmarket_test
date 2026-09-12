"""뉴스 기사에 인용된 증권사 목표주가 → 종목별 컨센서스.

데이터 경로
    NAVER API HUB 뉴스 검색 API (공식, 한시적 무료, 월 775,000건).
    네이버 금융·네이버 뉴스·한경컨센서스는 robots.txt가 전면 차단이라 쓰지 않는다.
    API가 주는 제목과 요약문(2~3줄)에서만 추출한다. 기사 본문은 받지 않는다.

이 모듈이 막는 함정 (첫 호출 10건에서 전부 실제로 나왔다)
    ① 복제 — 리포트 하나를 여러 매체가 받아쓴다. 삼성전자 첫 10건 중 5건이
       같은 KB증권 리포트였다. (증권사, 목표가)가 같으면 한 건으로 합친다.
    ② 오탐 — "삼성전기 139만원선", "삼성전자우 193,700원"은 목표가가 아니라
       다른 종목의 현재가다. '목표(주)가' 문구에 붙은 숫자만 채택하고,
       그 사이에 다른 종목명이 끼면 버린다.
    ③ 귀속 불명 — 증권사 이름이 없는 목표가는 누구 의견인지 몰라 버린다.

집계는 증권사당 최신 1건, 최근 90일. 목표가는 체계적으로 낙관적이라
절대값이 아니라 **종목 간 순위**로만 쓴다.
"""
from __future__ import annotations

import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config

ENDPOINT = "https://naverapihub.apigw.ntruss.com/search/v1/news"
MIN_INTERVAL = 0.25          # 초. 한도는 키당 50 RPS지만 급할 이유가 없다
WINDOW_DAYS = 90
DEDUPE_DAYS = 14             # 같은 (증권사, 목표가)가 이 안에 다시 나오면 같은 리포트
MIN_BROKERS = 3              # 이보다 적으면 중간값·분산을 계산하지 않는다

_last_call = 0.0
KST = timezone(timedelta(hours=9))

# ---------------------------------------------------------------------------
# 증권사 — 이름 목록으로 식별한다. "…증권" 정규식은 "증권가", "증권업계"까지 잡는다.
# 사명 변경은 현재 이름으로 합친다.
# ---------------------------------------------------------------------------
DOMESTIC = {
    "미래에셋증권": ["미래에셋증권", "미래에셋대우"],
    "삼성증권": ["삼성증권"],
    "NH투자증권": ["NH투자증권", "NH투자"],
    "KB증권": ["KB증권"],
    "한국투자증권": ["한국투자증권", "한투증권"],
    "키움증권": ["키움증권"],
    "신한투자증권": ["신한투자증권", "신한금융투자"],
    "하나증권": ["하나증권", "하나금융투자"],
    "메리츠증권": ["메리츠증권"],
    "대신증권": ["대신증권"],
    "유안타증권": ["유안타증권"],
    "교보증권": ["교보증권"],
    "한화투자증권": ["한화투자증권"],
    "현대차증권": ["현대차증권"],
    "DB증권": ["DB증권", "DB금융투자"],
    "iM증권": ["iM증권", "하이투자증권"],
    "IBK투자증권": ["IBK투자증권"],
    "SK증권": ["SK증권"],
    "유진투자증권": ["유진투자증권"],
    "LS증권": ["LS증권", "이베스트투자증권"],
    "상상인증권": ["상상인증권"],
    "신영증권": ["신영증권"],
    "부국증권": ["부국증권"],
    "한양증권": ["한양증권"],
    "케이프투자증권": ["케이프투자증권"],
    "BNK투자증권": ["BNK투자증권"],
    "다올투자증권": ["다올투자증권"],
    "흥국증권": ["흥국증권"],
    "리딩투자증권": ["리딩투자증권"],
    "DS투자증권": ["DS투자증권"],
    "우리투자증권": ["우리투자증권"],
}
FOREIGN = {
    "모건스탠리": ["모건스탠리", "모간스탠리"],
    "골드만삭스": ["골드만삭스"],
    "JP모건": ["JP모건", "JP모간", "제이피모건"],
    "맥쿼리": ["맥쿼리"],
    "CLSA": ["CLSA"],
    "노무라": ["노무라"],
    "UBS": ["UBS"],
    "씨티": ["씨티그룹", "씨티"],
    "HSBC": ["HSBC"],
    "제프리스": ["제프리스"],
    "BofA": ["뱅크오브아메리카", "BofA", "메릴린치"],
    "번스타인": ["번스타인"],
    "도이치": ["도이치은행", "도이치"],
    "바클레이즈": ["바클레이즈"],
}
# 기사 제목은 "하나證", "NH투자證"처럼 증권을 證으로 줄여 쓴다. 한국투자증권은 "한투".
for _c, _al in DOMESTIC.items():
    if _c.endswith("증권"):
        _al.append(_c[:-2] + "證")
    # "유진투자 "펄어비스 목표주가 상향…"", "신한證 "현대차…"" — 투자증권은
    # '증권'만 떼거나 '투자증권'을 통째로 證으로 줄여 쓴다
    if _c.endswith("투자증권"):
        _al.append(_c[:-2])
        _al.append(_c[:-4] + "證")
DOMESTIC["한국투자증권"].append("한투")

_ALIAS = [(a, canon, False) for canon, al in DOMESTIC.items() for a in al] + \
         [(a, canon, True) for canon, al in FOREIGN.items() for a in al]
# 긴 별칭부터 — "NH투자증권"이 "NH투자"보다 먼저 잡혀야 한다
_ALIAS.sort(key=lambda x: -len(x[0]))
_BROKER_RE = re.compile("|".join(re.escape(a) for a, _, _ in _ALIAS))
_CANON = {a: (c, f) for a, c, f in _ALIAS}

# ---------------------------------------------------------------------------
# 종목 약칭 — "삼전닉스" 기사가 삼성전자·SK하이닉스 목표가를 한 문장에 섞는다.
# 약칭을 모르면 "삼전 67만·닉스 470만"에서 누구 목표가인지 가릴 수 없다.
# ---------------------------------------------------------------------------
STOCK_ALIASES = {
    "삼성전자": ["삼전"],
    "SK하이닉스": ["하이닉스", "닉스", "SK하닉", "하닉"],
    "현대차": ["현대자동차", "현차"],
    "LG에너지솔루션": ["LG엔솔", "엔솔"],
    "삼성바이오로직스": ["삼성바이오", "삼바"],
    "KB금융": ["KB금융지주"],
    "신한지주": ["신한금융지주", "신한금융"],
    "하나금융지주": ["하나금융"],
    "우리금융지주": ["우리금융"],
    "한국금융지주": ["한국투자금융지주"],
    "메리츠금융지주": ["메리츠금융"],
    "HD현대중공업": ["현대중공업"],
    "NAVER": ["네이버"],
}

# 계열 증권사 — 금융지주 기사에서 "KB증권"은 애널리스트가 아니라 자회사 이름이다.
# 증권사는 계열사 분석 공표가 제한되므로, 이름이 보여도 애널리스트로 귀속하지 않는다.
AFFILIATE = {
    "KB금융": {"KB증권"}, "신한지주": {"신한투자증권"}, "하나금융지주": {"하나증권"},
    "우리금융지주": {"우리투자증권"}, "한국금융지주": {"한국투자증권"},
    "메리츠금융지주": {"메리츠증권"}, "BNK금융지주": {"BNK투자증권"},
    "iM금융지주": {"iM증권"}, "현대차": {"현대차증권"}, "한화": {"한화투자증권"},
    "DB손해보험": {"DB증권"}, "미래에셋증권": {"미래에셋증권"}, "삼성증권": {"삼성증권"},
    "NH투자증권": {"NH투자증권"}, "키움증권": {"키움증권"}, "대신증권": {"대신증권"},
}

# "9월 첫째주 삼성전자 등 7종목 매수 추천" — 여러 종목을 늘어놓는 기사는 통째로 버린다
_MULTI = re.compile(r"\d+\s*종목|종목\s*추천|주간\s*추천")


# 전체 상장사명 — 스크립트가 채운다. "CJ"가 "CJ제일제당" 안에서 잡히지 않게,
# 우리 종목명으로 시작하는 더 긴 상장사명을 부정 전방탐색에 넣는다.
# 유니버스(199)만 보면 유니버스 밖 자회사(CJ ENM, CJ프레시웨이…)를 못 막는다.
# 첫 전 종목 실행에서 CJ가 배율 탈락 28건·분산 53%로 튀었다.
MARKET_NAMES: list[str] = []


def name_patterns(name: str) -> list[re.Pattern]:
    out = []
    for n in [name, *STOCK_ALIASES.get(name, [])]:
        longer = sorted({m[len(n):] for m in MARKET_NAMES
                         if m.startswith(n) and len(m) > len(n)}, key=len, reverse=True)
        block = "|".join(["우", r"\d우", "우B", *map(re.escape, longer)])
        out.append(re.compile(re.escape(n) + f"(?!{block})"))
    return out


def other_patterns(name: str, universe: list[str]) -> list[re.Pattern]:
    """대상 종목 외 모든 종목명·약칭. 대상 이름에 포함되는 것(삼성전자 ⊂ 삼성전자우)은 뺀다."""
    mine = [name, *STOCK_ALIASES.get(name, [])]
    out = []
    for n in universe:
        if n == name:
            continue
        for a in [n, *STOCK_ALIASES.get(n, [])]:
            if len(a) >= 2 and not any(a in m for m in mine):
                out.append(_name_pattern(a))
    return out


# ---------------------------------------------------------------------------
# 금액 — "60만원", "12만5000원", "13만5천원", "1,230,000원", "7200원"
# ---------------------------------------------------------------------------
_PRICE = re.compile(
    r"(?P<a>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*"
    r"(?:(?P<man>만)\s*(?:(?P<b>\d{1,4}(?:,\d{3})?)\s*(?P<cheon>천)?)?)?\s*원")
_TARGET_KW = re.compile(r"목표\s*(?:주)?가(?:격)?")
# 순서가 중요하다 — "신규 제시"가 "제시"에 먼저 걸려 유지로 잡히지 않게 신규를 앞에 둔다
_DIR = [("상향", r"상향|올려|올렸|높여|높였|끌어올"),
        ("하향", r"하향|낮춰|낮췄|내려|내렸|낮아"),
        ("신규", r"신규|커버리지 개시|분석을 개시"),
        ("유지", r"유지|제시")]


def _price_value(m: re.Match) -> float | None:
    a = float(m.group("a").replace(",", ""))
    if m.group("man"):
        b = m.group("b")
        rest = 0.0
        if b:
            rest = float(b.replace(",", "")) * (1000 if m.group("cheon") else 1)
        return a * 10000 + rest
    return a


def clean(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).replace("​", "")


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+|\s*…\s*|\n", text)
    return [p.strip() for p in parts if p.strip()]


def _name_pattern(name: str) -> re.Pattern:
    # "삼성전자"가 "삼성전자우"에 걸리지 않게 우선주 접미사를 막는다
    return re.compile(re.escape(name) + r"(?!우|\d우|우B)")


# ---------------------------------------------------------------------------
# 추출
# ---------------------------------------------------------------------------
def extract(item: dict, name: str, others: list[re.Pattern]) -> list[dict]:
    """기사 1건 → 목표가 관측치 0~n개.

    귀속 규칙 — 정확도를 위해 재현율을 희생한다. 첫 테스트에서 삼성화재·삼성물산·
    삼성E&A 기사가 "삼성전자"를 언급하며 자기 목표가를 냈고, 그게 삼성전자 목표가로
    잡혔다.
      · 목표가 문구와 **같은 문장**에 대상 종목명(또는 약칭)이 있어야 한다
      · 그 문장에 **다른 종목명이 하나라도** 있으면 버린다
      · 여러 종목을 늘어놓는 기사("7종목 추천")는 통째로 버린다
      · 대상 종목의 계열 증권사 이름은 애널리스트로 보지 않는다

    others : other_patterns()의 결과
    """
    title, desc = clean(item.get("title", "")), clean(item.get("description", ""))
    if _MULTI.search(title):
        return []
    mine = name_patterns(name)

    def mask(s: str) -> str:
        # 증권사 이름을 지운 뒤에 종목명을 찾는다. 키움증권·NH투자증권·미래에셋증권은
        # 그 자체가 상장사이고, LS증권·한화투자증권·현대차증권은 이름 안에 상장사명
        # (LS·한화·현대차)이 들어 있다. 안 지우면 "키움증권은 KB금융 목표주가를 24만원으로
        # 상향"이 '다른 종목이 낀 문장'으로 탈락한다 — 실제로 KB금융에서 복제 2건 이상인
        # 진짜 발표 8건이 이렇게 빠져 중간값이 네이버 컨센서스보다 6.9% 낮게 나왔다.
        return _BROKER_RE.sub(lambda m: m.group(0) if m.group(0) == name else "〔증권사〕", s)

    # 제목이 **다른 종목에 관한** 기사는 버린다. "노무라 '삼전 목표가 67만원'" 기사의
    # 요약문에서 현대차 목표가가 뽑힌 적이 있다. 제목에 대상 종목이 없고 다른 종목만
    # 있으면 그 기사의 주인공은 우리 종목이 아니다.
    tm = mask(title)
    if not any(m.search(tm) for m in mine) and any(o.search(tm) for o in others):
        return []
    try:
        pub = parsedate_to_datetime(item["pubDate"]).astimezone(KST)
    except Exception:
        return []
    skip_brokers = AFFILIATE.get(name, set())
    out = []
    for si, sent in enumerate([title] + _sentences(desc)):
        if not any(m.search(mask(sent)) for m in mine):
            continue
        for kw in _TARGET_KW.finditer(sent):
            # 다른 종목명은 목표가 문구 **앞쪽**에서만 본다. 앞에 있으면 그 종목이 주어일
            # 수 있다(삼성화재·삼성물산 기사). 뒤에만 있으면 "삼전 43만·하닉 300만"처럼
            # 첫 금액이 우리 종목 것이다.
            if any(o.search(mask(sent[: kw.start()])) for o in others):
                continue
            # 목표가 문구 뒤 60자 안의 금액들
            win = sent[kw.end(): kw.end() + 60]
            prices = list(_PRICE.finditer(win))
            if not prices:
                continue
            # "A원에서 B원으로" → 새 목표가는 B, 이전은 A
            tgt, prev = prices[0], None
            if len(prices) >= 2 and "에서" in win[prices[0].end(): prices[1].start()]:
                prev, tgt = prices[0], prices[1]
            value = _price_value(tgt)
            if not value or value < 1000:
                continue

            # 증권사 — 목표가 문구 앞쪽의 가장 가까운 이름, 없으면 문장 → 제목
            ok = lambda m: _CANON[m.group(0)][0] not in skip_brokers  # noqa: E731
            cands = [m for m in _BROKER_RE.finditer(sent[: kw.start()]) if ok(m)]
            if not cands:
                cands = [m for m in [*_BROKER_RE.finditer(sent), *_BROKER_RE.finditer(title)]
                         if ok(m)]
            if not cands:
                continue
            canon, foreign = _CANON[cands[-1].group(0)]

            tail = sent[kw.start(): kw.end() + 80]
            direction = next((d for d, pat in _DIR if re.search(pat, tail)), None)
            out.append({
                "broker": canon, "foreign": foreign, "target": value,
                "prev_target": _price_value(prev) if prev else None,
                "direction": direction, "pub": pub,
                "title": title[:80], "link": item.get("originallink") or item.get("link"),
                "from_title": si == 0,
            })
    # 같은 기사에서 제목이 준 (증권사, 목표가)가 있으면 본문의 다른 값은 버린다.
    # 신한證 "현대차…목표가 78만원으로 하향" 기사에서 본문 요약의 80만원(이전 값)이
    # 따로 잡힌 적이 있다. 제목은 편집자가 요약한 결론이라 더 믿을 만하다.
    title_says = {o["broker"]: round(o["target"]) for o in out if o["from_title"]}
    out = [o for o in out if o["from_title"] or o["broker"] not in title_says
           or round(o["target"]) == title_says[o["broker"]]]
    # 한 기사 안에서 같은 (증권사, 목표가)가 제목·본문에 두 번 나오면 하나로
    seen, uniq = set(), []
    for o in out:
        k = (o["broker"], round(o["target"]))
        if k not in seen:
            seen.add(k)
            uniq.append(o)
    return uniq


def dedupe(obs: pd.DataFrame) -> pd.DataFrame:
    """복제 기사 합치기 — 같은 증권사가 같은 목표가를 DEDUPE_DAYS 안에 또 내면 같은 리포트."""
    if obs.empty:
        return obs
    obs = obs.sort_values("pub")
    keep, last = [], {}
    for i, r in obs.iterrows():
        k = (r["broker"], round(r["target"]))
        if k in last and (r["pub"] - last[k]).days <= DEDUPE_DAYS:
            continue
        last[k] = r["pub"]
        keep.append(i)
    d = obs.loc[keep].copy()
    d["copies"] = [int(((obs["broker"] == r.broker) & (obs["target"].round() == round(r.target))
                        & ((obs["pub"] - r.pub).dt.days.between(0, DEDUPE_DAYS))).sum())
                   for r in d.itertuples()]
    return d


def representatives(obs: pd.DataFrame, asof: datetime,
                    domestic_only: bool = True) -> pd.DataFrame:
    """증권사별 대표 목표가 1건 (최근 WINDOW_DAYS일).

    여러 매체에 실린(복제 2건 이상) 발표 중 최신을 우선한다. 나중 기사가 옛 목표가를
    다시 인용하는 일이 흔하다: KB증권은 현대차 목표가를 7/21에 120만→90만으로
    낮췄는데(기사 10건), 8/10 "목표가 극과 극" 기사가 옛 120만원을 언급해 그게
    최신으로 잡혔다. 실제 발표는 당일 여러 곳에 실리고 재인용은 한 번뿐이다.
    복제 2건 이상이 없는 증권사만 1건짜리로 채운다.
    """
    if obs.empty:
        return obs
    w = obs[obs["pub"] >= asof - timedelta(days=WINDOW_DAYS)]
    if domestic_only:
        w = w[~w["foreign"]]
    if w.empty:
        return w
    if "copies" not in w:
        return w.sort_values("pub").groupby("broker").tail(1)

    picks = []
    for _, g in w.groupby("broker"):
        g = g.sort_values("pub")
        conf = g[g["copies"] >= 2]
        if not len(conf):
            picks.append(g.index[-1])
            continue
        cur = conf.iloc[-1]
        # 확정 발표 이후의 1건짜리 관측은 대개 옛 목표가 재인용이거나 오추출이다.
        # 다만 **방향 라벨이 값의 변화와 맞으면** 진짜 갱신으로 본다.
        # 실측: 복제 1건 관측의 방향 모순율 31% vs 복제 2건+ 14%. 밀려 있던 96건 중
        # 방향 일치는 36건이었고, 삼성전기를 NH·신한이 나란히 300만→220만으로 낮춘
        # 것처럼 실제 하향이었다. 나머지(모순 22·방향없음 38)는 확정 발표를 유지한다.
        for _, r in g[g["pub"] > cur["pub"]].iterrows():
            chg = r["target"] / cur["target"] - 1
            if (r["direction"] in ("상향", "하향") and abs(chg) > 1e-9
                    and (r["direction"] == "상향") == (chg > 0)):
                cur = r
        picks.append(cur.name)
    return w.loc[picks]


def aggregate(obs: pd.DataFrame, price: float | None, asof: datetime,
              domestic_only: bool = True) -> dict:
    """증권사별 대표값 → 5개 값."""
    empty = {"n_brokers": 0, "median_target": None, "upside": None,
             "dispersion": None, "revision": None, "n_foreign": 0}
    if obs.empty:
        return empty
    w = obs[obs["pub"] >= asof - timedelta(days=WINDOW_DAYS)]
    n_foreign = int(w[w["foreign"]]["broker"].nunique())
    latest = representatives(obs, asof, domestic_only)
    if latest.empty:
        return {**empty, "n_foreign": n_foreign}
    n = len(latest)
    res = {**empty, "n_brokers": n, "n_foreign": n_foreign}
    if n < MIN_BROKERS:
        return res
    med = float(latest["target"].median())
    q1, q3 = latest["target"].quantile([0.25, 0.75])
    ups = latest["direction"].eq("상향").sum()
    downs = latest["direction"].eq("하향").sum()
    res.update({
        "median_target": med,
        "upside": (med / price - 1) * 100 if price else None,
        "dispersion": float((q3 - q1) / med * 100),      # 사분위 폭 / 중간값 (%)
        "revision": float((ups - downs) / n),
    })
    return res


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def _raw_path(query: str, start: int, day: str) -> Path:
    safe = re.sub(r"[^\w가-힣]+", "_", query)
    return config.RAW / "naver" / "news" / day / f"{safe}_{start}.json"


def search(query: str, *, since: datetime, max_pages: int = 10,
           cache_day: str | None = None) -> list[dict]:
    """날짜순으로 받다가 `since`보다 오래된 기사가 나오면 멈춘다.

    원본 응답은 data/raw/naver/news/{날짜}/ 에 남긴다 — 추출 규칙을 고쳐도
    다시 호출하지 않고 재파싱하기 위해서다 (DART 원본을 남기는 것과 같은 이유).
    """
    global _last_call
    cid, sec = os.environ.get("NAVER_CLIENT_ID"), os.environ.get("NAVER_CLIENT_SECRET")
    if not cid or not sec:
        raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 이 .env 에 없습니다")
    day = cache_day or datetime.now(KST).strftime("%Y%m%d")
    items: list[dict] = []
    for page in range(max_pages):
        start = 1 + page * 100
        if start > 1000:
            break
        p = _raw_path(query, start, day)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            wait = MIN_INTERVAL - (time.monotonic() - _last_call)
            if wait > 0:
                time.sleep(wait)
            q = urllib.parse.urlencode({"query": query, "display": 100,
                                        "start": start, "sort": "date"})
            req = urllib.request.Request(f"{ENDPOINT}?{q}", headers={
                "X-NCP-APIGW-API-KEY-ID": cid, "X-NCP-APIGW-API-KEY": sec})
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    data = json.load(r)
            except urllib.error.HTTPError as e:
                if e.code == 429:          # 초당 한도 — 1초 쉬고 한 번만 재시도
                    time.sleep(1.2)
                    with urllib.request.urlopen(req, timeout=20) as r:
                        data = json.load(r)
                else:
                    raise
            finally:
                _last_call = time.monotonic()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        batch = data.get("items", [])
        items.extend(batch)
        if not batch:
            break
        try:
            oldest = parsedate_to_datetime(batch[-1]["pubDate"])
        except Exception:
            break
        if oldest < since or len(batch) < 100:
            break
    return items


def cached_items(query: str) -> list[dict]:
    """이 검색어로 지금까지 받아둔 원본 전부(모든 날짜). 같은 기사는 링크로 한 번만.

    관측치는 저장해 두지 않고 매번 원본에서 다시 뽑는다. 이력을 누적 저장하면
    옛 규칙으로 잘못 뽑은 값이 규칙을 고친 뒤에도 남는다 — 원본이 기준이다.
    """
    safe = re.sub(r"[^\w가-힣]+", "_", query)
    seen, out = set(), []
    for p in sorted((config.RAW / "naver" / "news").glob(f"*/{safe}_*.json")):
        try:
            items = json.loads(p.read_text(encoding="utf-8")).get("items", [])
        except Exception:
            continue
        for it in items:
            k = it.get("originallink") or it.get("link")
            if k and k not in seen:
                seen.add(k)
                out.append(it)
    return out


def api_calls_in_cache(day: str) -> int:
    d = config.RAW / "naver" / "news" / day
    return len(list(d.glob("*.json"))) if d.exists() else 0
