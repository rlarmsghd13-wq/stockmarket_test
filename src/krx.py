"""KRX / pykrx 래퍼.

- 모든 호출은 디스크 캐시를 거친다. KRX는 느리고 로그인 세션이 1시간마다 만료된다.
- 수정주가만 쓴다. 액면분할·무상증자 미반영 주가는 성장률과 수익률을 전부 망친다.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import pandas as pd

import config
from . import env

env.load()

# pykrx는 **import 시점에 KRX 로그인을 시도**한다. 차단 상태에서 모듈을 최상단에서
# 임포트하면 `from src import krx` 자체가 예외로 죽어, 상태를 확인하는 코드조차
# 실행되지 않는다. 실제로 쓸 때 불러오도록 미룬다.
_stock = None


def _api():
    global _stock
    if _stock is None:
        from pykrx import stock as _s
        _stock = _s
    return _stock


def available() -> tuple[bool, str]:
    """KRX가 지금 응답하는가. 차단 여부 확인용 — 호출 1회만 쓴다."""
    try:
        t = _api().get_market_ticker_list("20260904", market="KOSPI")
        return (True, f"정상 — KOSPI {len(t)}종목 응답") if t else (False, "빈 응답")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:80]}"

_CACHE = config.RAW / "krx"
_CACHE.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 호출 속도 제한
#
# 2026-09-06에 KRX가 이 IP를 차단했다. 원인은 명확하다 —
#   · 하루 1,100회 이상을 간격 없이 최대 속도로 호출
#   · 시세 수집과 유니버스 복원을 **동시에** 실행해 세션 두 개가 각각 로그인
# 캐시가 있으면 네트워크를 타지 않으므로, 실제로 느려지는 건 첫 수집뿐이다.
# ---------------------------------------------------------------------------
MIN_INTERVAL = 0.35          # 초. 초당 약 3회
DAILY_SOFT_LIMIT = 800       # 넘으면 경고. 하드 차단은 하지 않는다
_last_call = 0.0
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
    _last_call = time.monotonic()
    _calls_today += 1
    if _calls_today == DAILY_SOFT_LIMIT:
        print(f"  [경고] KRX 호출 {_calls_today}회 — 오늘은 여기서 멈추는 것을 권합니다.",
              flush=True)


def call_count() -> int:
    return _calls_today


def _cached(name: str, fn, *, retries: int = 3, cache_empty: bool = True):
    p = _CACHE / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
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
        raise RuntimeError(f"KRX 호출 실패: {name}") from last
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

    return _cached(f"cap_{market}_{d}", _fetch)


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

    return _cached(f"names_{market}_{d}", _fetch)


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
