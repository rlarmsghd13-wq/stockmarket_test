"""KRX / pykrx 래퍼.

- 모든 호출은 디스크 캐시를 거친다. KRX는 느리고 로그인 세션이 1시간마다 만료된다.
- 수정주가만 쓴다. 액면분할·무상증자 미반영 주가는 성장률과 수익률을 전부 망친다.

**2026-09-27부터 웹 수집을 멈췄다 — 캐시 조회만 된다.** 아래 SCRAPING_ALLOWED 참조.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import pandas as pd

import config
from . import env

env.load()

# ---------------------------------------------------------------------------
# 웹 수집 중단 — 공식 Open API로 옮길 때까지
#
# 2026-09-26 차단 안내문에 이렇게 적혀 있었다.
#
#     "접속 제한 해제 후에는 정상적으로 이용하실 수 있습니다.
#      다만 자동화 수단을 통한 데이터 수집은 제한되며 …"
#
# KRX는 빠르게 긁는 것만이 아니라 **웹사이트를 자동으로 긁는 것 자체**를
# 이용약관(제10조 제2호)으로 제한한다. pykrx가 정확히 그 방식이다 — 개인 KRX
# 아이디로 로그인한 세션으로 data.krx.co.kr을 긁는다. 속도를 늦춘 것은 걸릴
# 확률을 낮췄을 뿐 약관 문제를 풀지 않는다.
#
# 그래서 KRX가 자동 수집용으로 따로 열어둔 **공식 Open API(openapi.krx.co.kr)**
# 로 옮기기로 했다. 한경컨센서스·네이버 뉴스를 robots.txt 때문에 공식 API로
# 옮긴 것과 같은 결정이다.
#
# 옮길 때까지 캐시에 있는 데이터(시세 96만 행, 스냅샷 30개)는 그대로 쓰고,
# **캐시에 없는 요청은 네트워크에 나가지 않고 즉시 거절한다.**
# 되돌리려면 이 값을 사람이 직접 True로 바꿔야 한다 — 일부러 번거롭게 뒀다.
# ---------------------------------------------------------------------------
SCRAPING_ALLOWED = False
SCRAPING_STOPPED_MSG = (
    "KRX 웹 수집은 2026-09-27부터 멈췄습니다 (약관상 자동 수집 제한). "
    "공식 Open API(openapi.krx.co.kr) 전환을 기다리는 중이라 캐시에 있는 "
    "데이터만 쓸 수 있습니다. src/krx.py의 SCRAPING_ALLOWED 설명을 보세요.")

# pykrx는 **import 시점에 KRX 로그인을 시도**한다. 차단 상태에서 모듈을 최상단에서
# 임포트하면 `from src import krx` 자체가 예외로 죽어, 상태를 확인하는 코드조차
# 실행되지 않는다. 실제로 쓸 때 불러오도록 미룬다.
_stock = None


def _api():
    global _stock
    if not SCRAPING_ALLOWED:
        # import 자체가 로그인이므로 여기서 막아야 한다
        raise Blocked(SCRAPING_STOPPED_MSG)
    if _stock is None:
        from pykrx import stock as _s
        _stock = _s
    return _stock


BLOCK_MARK = "이용 제한"      # KRX 차단 안내 페이지에 들어 있는 문구


def block_notice(timeout: float = 15.0) -> str | None:
    """차단 안내 페이지를 읽어 사람이 알아볼 수 있는 사유를 돌려준다.

    차단 중에는 pykrx가 `JSONDecodeError: Expecting value: line 13 column 1`을
    낸다. 그 메시지만 보면 날짜가 잘못됐나 싶지만, 실제로는
    data.krx.co.kr **전체**가 아래 안내 페이지를 준다.

        "자동화 수단을 통한 비정상 대량 조회가 감지되어 해당 IP의 접속이
         일시적으로 제한되었습니다 … 탐지일로부터 1일간"

    로그인 요청이 아니라 평범한 GET 1회만 쓴다. 차단을 확인하는 행위가
    다시 '대량 조회'로 잡히지 않게, 실패 경로에서만 부른다.
    """
    import re
    import requests
    try:
        r = requests.get(
            "https://data.krx.co.kr/contents/MDC/COMS/client/MDCCOMS001.cmd",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    except Exception:
        return None
    if BLOCK_MARK not in r.text:
        return None
    txt = re.sub(r"<(script|style).*?</\1>", " ", r.text, flags=re.S)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    i = txt.find(BLOCK_MARK)
    return txt[i:i + 220]


def available() -> tuple[bool, str]:
    """KRX가 지금 응답하는가. 차단 여부 확인용 — 호출 1회만 쓴다."""
    global _circuit_open
    if not SCRAPING_ALLOWED:
        # 상태 확인도 로그인이고, 실패하면 block_notice()가 또 GET을 한다.
        # 수집을 멈춘 동안에는 아무 요청도 보내지 않는다.
        return False, "웹 수집 중단 — 공식 Open API 전환 대기 (src/krx.py)"
    try:
        t = _api().get_market_ticker_list("20260904", market="KOSPI")
        if t:
            return True, f"정상 — KOSPI {len(t)}종목 응답"
        why = "빈 응답"
    except Exception as exc:
        why = f"{type(exc).__name__}: {str(exc)[:80]}"
    notice = block_notice()
    _circuit_open = True          # 차단 확인 — 더 두드리지 않는다
    if notice:
        return False, f"IP 차단됨 — {notice}"
    return False, why


_CACHE = config.RAW / "krx"
_CACHE.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 호출 속도 제한
#
# 두 번 차단당하고 알게 된 것: **문제는 하루 총량이 아니라 순간 속도다.**
#   · 2026-09-06 — 하루 1,100회 이상을 간격 없이, 그리고 두 작업을 동시에 실행
#   · 2026-09-26 — 하루 **190회**뿐이었는데도 차단됐다. 0.35초 간격으로
#     90초 동안 162회를 몰아친 것이 "자동화 수단을 통한 비정상 대량 조회"로
#     잡혔다. 그때 하루 한도 800회는 4분의 1도 쓰지 않은 상태였다.
#
# 그래서 간격을 늘리고, 짧은 창 안의 건수도 따로 막는다. 캐시가 있으면
# 네트워크를 타지 않으므로 느려지는 건 첫 수집뿐이다.
# 스냅샷 16개(약 290콜)는 이 설정으로 약 15분 걸린다 — 월 1회 작업에 충분하다.
# ---------------------------------------------------------------------------
MIN_INTERVAL = 2.0           # 초. 0.35초가 차단을 불렀다
BURST_WINDOW = 300           # 초
BURST_LIMIT = 100            # 이 창 안에서 이만큼 넘으면 창이 빌 때까지 기다린다
DAILY_SOFT_LIMIT = 400       # 넘으면 경고. 하드 차단은 하지 않는다
_last_call = 0.0
_recent: list[float] = []    # 최근 호출 시각 (BURST_WINDOW 안)
_calls_today = 0
_call_day = date.today()


def _throttle() -> None:
    global _last_call, _calls_today, _call_day
    today = date.today()
    if today != _call_day:
        _call_day, _calls_today = today, 0

    wait = MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)

    # 창 안의 건수 제한 — 평균 속도가 아니라 순간 속도를 막는 장치다
    now = time.monotonic()
    _recent[:] = [t for t in _recent if now - t < BURST_WINDOW]
    if len(_recent) >= BURST_LIMIT:
        rest = BURST_WINDOW - (now - _recent[0]) + 1
        if rest > 0:
            print(f"  [속도] 최근 {BURST_WINDOW}초에 {len(_recent)}회 — "
                  f"{rest:.0f}초 쉽니다.", flush=True)
            time.sleep(rest)
        now = time.monotonic()
        _recent[:] = [t for t in _recent if now - t < BURST_WINDOW]

    _last_call = now
    _recent.append(now)
    _calls_today += 1
    if _calls_today == DAILY_SOFT_LIMIT:
        print(f"  [경고] KRX 호출 {_calls_today}회 — 오늘은 여기서 멈추는 것을 권합니다.",
              flush=True)


def call_count() -> int:
    return _calls_today


# ---------------------------------------------------------------------------
# 차단기 — 연속 실패가 쌓이면 네트워크를 아예 끊는다
#
# pykrx는 요청이 실패하면 **로그인을 다시 시도한다.** 그래서 `_cached`가
# 재시도 3회를 돌면 실패한 호출 하나마다 로그인 시도가 3번 발생한다.
# 2026-09-26에 KRX 로그인 엔드포인트가 JSON 대신 HTML을 주기 시작했는데,
# 스냅샷 6개가 각각 3번씩 재시도해 짧은 시간에 로그인 실패가 18번 쌓였다.
# 이것이 정확히 과거 24시간 IP 차단을 불렀던 행동이다.
#
# 연속 실패가 CIRCUIT_TRIP에 닿으면 이후 모든 호출을 네트워크 없이 즉시
# 거절한다. 캐시 조회는 계속 동작한다.
# ---------------------------------------------------------------------------
CIRCUIT_TRIP = 2
_consec_fail = 0
_circuit_open = False


class Blocked(RuntimeError):
    """KRX가 응답하지 않아 차단기가 열렸다. 재시도하지 말고 나중에 다시."""


def circuit_open() -> bool:
    return _circuit_open


def reset_circuit() -> None:
    """사람이 상태를 확인한 뒤에만 부른다."""
    global _consec_fail, _circuit_open
    _consec_fail, _circuit_open = 0, False


def _cached(name: str, fn, *, retries: int = 2, cache_empty: bool = True):
    global _consec_fail, _circuit_open
    p = _CACHE / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    if not SCRAPING_ALLOWED:
        # 재시도·대기 없이 바로 거절한다. 캐시 적중은 위에서 이미 돌려줬다.
        raise Blocked(f"{name}: 캐시에 없음. {SCRAPING_STOPPED_MSG}")
    if _circuit_open:
        raise Blocked(
            f"KRX 차단기가 열려 있어 {name}을 요청하지 않았습니다. "
            "연속 실패가 누적됐습니다 — 몇 시간 뒤 다시 실행하세요.")
    last = None
    for i in range(retries):
        try:
            _throttle()
            df = fn()
            break
        except Exception as exc:          # KRX 세션 만료 / 일시 오류
            last = exc
            # 실패 시 물러서는 간격을 크게 잡는다. 차단 상태에서 빠르게
            # 재시도하면 차단이 길어질 뿐이다.
            time.sleep(3.0 * (i + 1))
    else:
        _consec_fail += 1
        if _consec_fail >= CIRCUIT_TRIP:
            _circuit_open = True
            print(f"  [차단기] KRX 연속 실패 {_consec_fail}회 — 이후 호출을 "
                  "중단합니다. 로그인 실패를 반복하면 IP가 차단됩니다.",
                  flush=True)
        raise RuntimeError(f"KRX 호출 실패: {name}") from last
    _consec_fail = 0
    if df is None or df.empty:
        df = pd.DataFrame()
    if cache_empty or not df.empty:
        df.to_parquet(p)
    return df


def is_settled(snap: pd.DataFrame) -> bool:
    """KRX는 휴장일과 '아직 확정 전'인 당일에 행은 주되 값을 전부 0으로 채운다.

    비어 있는지만 보면 0으로 가득한 스냅샷을 유효한 영업일로 착각하게 된다.
    실제로 값이 들어찼는지로 판정한다.
    """
    if snap is None or snap.empty or "close" not in snap.columns:
        return False
    return bool((snap["close"] > 0).sum() > len(snap) * 0.5)


def ymd(d) -> str:
    if isinstance(d, str):
        return d.replace("-", "")[:8]
    return d.strftime("%Y%m%d")


# ---------------------------------------------------------------------------
# 리밸런싱 날짜 — 각 보고서 제출 마감 직후
# ---------------------------------------------------------------------------
def rebalance_dates(start: str = config.BACKTEST_START,
                    end: str | None = None) -> list[str]:
    """사업보고서(3/31) → 4/05, 1Q(5/15) → 5/20, 반기(8/14) → 8/20, 3Q(11/14) → 11/20.

    공시 마감 직후로 잡아야 그 시점에 실제로 볼 수 있었던 재무로 판단하게 된다.
    """
    end_d = date.fromisoformat(end) if end else date.today()
    start_d = date.fromisoformat(start)
    out: list[str] = []
    for y in range(start_d.year, end_d.year + 1):
        for mm, dd in ((4, 5), (5, 20), (8, 20), (11, 20)):
            d = date(y, mm, dd)
            if start_d <= d <= end_d:
                out.append(d.isoformat())
    return out


def prev_business_day(d: str, max_back: int = 12) -> str:
    """휴장일·미확정일이면 직전 영업일로."""
    cur = date.fromisoformat(d) if "-" in d else datetime.strptime(d, "%Y%m%d").date()
    for _ in range(max_back):
        if is_settled(market_snapshot(ymd(cur), "KOSPI")):
            return ymd(cur)
        cur -= timedelta(days=1)
    raise RuntimeError(f"{d} 기준 영업일을 찾지 못했습니다.")


# ---------------------------------------------------------------------------
# 단일 날짜 전 종목 스냅샷
# ---------------------------------------------------------------------------
def market_snapshot(d: str, market: str) -> pd.DataFrame:
    """그 날짜의 전 종목 종가·시총·거래량·거래대금·상장주식수."""
    d = ymd(d)

    def _fetch():
        df = _api().get_market_cap(d, market=market)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(columns={
            "티커": "ticker", "종가": "close", "시가총액": "market_cap",
            "거래량": "volume", "거래대금": "trading_value",
            "상장주식수": "shares_out",
        })
        df["market"] = market
        df["date"] = d
        return df

    # 전 종목 조회가 **빈 응답**이면 그건 답이 아니라 오류다. 캐시에 남기면
    # 그 날짜가 영구히 망가진다 — names_KOSPI_20201120이 실제로 그래서
    # 2020-11 스냅샷을 8개월간 못 만들고 있었다.
    return _cached(f"cap_{market}_{d}", _fetch, cache_empty=False)


def fundamental_snapshot(d: str, market: str) -> pd.DataFrame:
    """KRX 제공 BPS/PER/PBR/EPS/DIV/DPS.

    주의 — 적자 기업의 PER을 0.00으로 준다. 우리 지표는 DART 재무로 직접 계산하고,
    이 값은 교차검증에만 쓴다. 그대로 쓰면 적자 기업이 '가장 싼 종목'이 된다.
    """
    d = ymd(d)

    def _fetch():
        df = _api().get_market_fundamental(d, market=market)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(columns={"티커": "ticker"})
        df.columns = [c.lower() if c != "ticker" else c for c in df.columns]
        df["market"] = market
        df["date"] = d
        return df

    return _cached(f"fund_{market}_{d}", _fetch)


def ticker_names(d: str, market: str) -> pd.DataFrame:
    d = ymd(d)

    def _fetch():
        tickers = _api().get_market_ticker_list(d, market=market)
        rows = [{"ticker": t, "name": _api().get_market_ticker_name(t)} for t in tickers]
        return pd.DataFrame(rows)

    return _cached(f"names_{market}_{d}", _fetch, cache_empty=False)


# ---------------------------------------------------------------------------
# 종목별 시계열
# ---------------------------------------------------------------------------
def ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """수정주가 일봉."""
    def _fetch():
        df = _api().get_market_ohlcv(ymd(start), ymd(end), ticker, adjusted=True)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(columns={
            "날짜": "date", "시가": "open", "고가": "high", "저가": "low",
            "종가": "close_adj", "거래량": "volume", "거래대금": "trading_value",
            "등락률": "change_pct",
        })
        df["ticker"] = ticker
        return df

    return _cached(f"ohlcv_{ticker}_{ymd(start)}_{ymd(end)}", _fetch)


def _flows_window(ticker: str, start: str, end: str) -> pd.DataFrame:
    def _fetch():
        df = _api().get_market_trading_value_by_date(ymd(start), ymd(end), ticker)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(columns={"날짜": "date"})
        ren = {"외국인합계": "foreign_net", "기관합계": "inst_net", "개인": "retail_net"}
        df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
        df["ticker"] = ticker
        keep = ["ticker", "date"] + [c for c in ("foreign_net", "inst_net", "retail_net")
                                     if c in df.columns]
        return df[keep]

    return _cached(f"flows_{ticker}_{ymd(start)}_{ymd(end)}", _fetch)


def investor_flows(ticker: str, start: str, end: str,
                   chunk_days: int = 300) -> pd.DataFrame:
    """투자자별 순매수 금액 — 급등 전략의 수급 지표.

    **연 단위로 쪼개서 받는다.** KRX는 이 엔드포인트에서 긴 구간을 거부한다.
    실측: 2년 창은 200종목 전부 성공했지만 4년 창은 65종목만 돌아왔다.
    한 덩어리로 요청하면 조용히 빈 값이 오므로 결측을 눈치채기 어렵다.
    """
    s = date.fromisoformat(start) if "-" in start else datetime.strptime(start, "%Y%m%d").date()
    e = date.fromisoformat(end) if "-" in end else datetime.strptime(end, "%Y%m%d").date()

    frames, cur = [], s
    while cur <= e:
        nxt = min(cur + timedelta(days=chunk_days), e)
        try:
            part = _flows_window(ticker, cur.isoformat(), nxt.isoformat())
            if not part.empty:
                frames.append(part)
        except Exception:
            pass          # 한 구간 실패가 전체를 날리지 않게 한다
        cur = nxt + timedelta(days=1)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["ticker", "date"]).sort_values("date")
