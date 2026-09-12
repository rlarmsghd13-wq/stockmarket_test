"""L2 — 파생 지표.

규칙
  - 분모가 없거나 0 이하면 결과는 None. 절대 0이나 음수로 채우지 않는다.
  - 적자 기업의 PER은 None. 0으로 두면 랭킹에서 '가장 싼 종목' 1위가 된다.
  - 트랙별로 계산하지 않는 지표는 config.TRACK_NULL_METRICS를 따라 None으로 만든다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import config



# 시가총액(원)과 재무제표 값을 함께 쓰는 지표들 — 표시 통화가 원화가 아니면 성립하지 않는다
MIXED_UNIT_METRICS = (
    "per", "pbr", "psr", "ev", "ev_ebit", "ev_ebitda",
    "eps", "bps", "fcf_yield", "peg", "normalized_per",
)

def _num(x):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def div(num, den, *, den_must_be_positive: bool = True):
    """안전 나눗셈. 분모가 없거나 부호가 맞지 않으면 None."""
    n, d = _num(num), _num(den)
    if n is None or d is None or d == 0:
        return None
    if den_must_be_positive and d < 0:
        return None
    return n / d


def effective_tax_rate(tax, pretax):
    r = div(tax, pretax)
    if r is None or not (config.TAX_RATE_BOUNDS[0] <= r <= config.TAX_RATE_BOUNDS[1]):
        return config.DEFAULT_TAX_RATE
    return r


def compute(row: dict, *, market_cap=None, shares_out=None, price=None,
            hist: pd.DataFrame | None = None, track: str = config.TRACK_GENERAL) -> dict:
    """한 (종목, 시점)의 지표 전부.

    row  : ttm.build() 한 행 (dict)
    hist : 그 시점까지의 TTM 시계열 (성장률·CAGR·정상화이익 계산용)
    """
    g = row.get
    m: dict[str, float | None] = {}

    rev = _num(g("revenue_ttm"))
    op = _num(g("operating_income_ttm"))
    ni = _num(g("net_income_ttm"))
    nip = _num(g("net_income_parent_ttm")) or ni
    ocf = _num(g("ocf_ttm"))
    fcf = _num(g("fcf_ttm"))
    da = _num(g("da_ttm"))
    gp = _num(g("gross_profit_ttm"))

    assets = _num(g("assets"))
    liab = _num(g("liabilities"))
    eq = _num(g("equity"))
    eqp = _num(g("equity_parent")) or eq
    ca = _num(g("current_assets"))
    cl = _num(g("current_liabilities"))
    cash = _num(g("cash"))
    eq_avg = _num(g("equity_avg")) or eq
    eqp_avg = _num(g("equity_parent_avg")) or eqp
    assets_avg = _num(g("assets_avg")) or assets

    mcap = _num(market_cap)
    tax_rate = effective_tax_rate(g("tax_expense_ttm"), g("pretax_income_ttm"))

    # ---------------- 수익성 ----------------
    m["roe"] = _pct(div(nip, eqp_avg))
    m["roa"] = _pct(div(ni, assets_avg))
    m["operating_margin"] = _pct(div(op, rev))
    m["net_margin"] = _pct(div(ni, rev))
    m["gp_to_assets"] = _pct(div(gp, assets))

    net_debt = None
    if eq is not None and liab is not None:
        borrowings = liab
        net_debt = borrowings - (cash or 0)
    m["net_debt"] = net_debt

    invested = None
    if eq is not None and net_debt is not None:
        invested = eq + max(net_debt, 0)
    m["roic"] = _pct(div((op * (1 - tax_rate)) if op is not None else None, invested))

    # ---------------- 안정성 ----------------
    m["debt_ratio"] = _pct(div(liab, eq))
    m["current_ratio"] = _pct(div(ca, cl))
    # 자기자본비율 — 금융업(T2) 안정성의 주 지표.
    # 은행·보험에는 부채비율이 의미가 없다(1,200%가 정상). BIS비율·K-ICS는
    # DART 표준 재무제표(XBRL 계정)에 없고 사업보고서 본문에만 있어 받을 수 없다.
    # 받을 수 있는 것 중 지급여력을 대리하는 유일한 지표가 이것이다.
    m["equity_ratio"] = _pct(div(eq, assets))
    m["ocf_to_ni"] = _pct(div(ocf, nip))
    m["fcf_to_ni"] = _pct(div(fcf, nip))

    # DART 요약 재무제표에는 감가상각비가 거의 없다 — 현금흐름표 '조정' 세부항목이라
    # fnlttSinglAcntAll이 내려주지 않는다 (실측 결측률 79%: 기아·NAVER는 계정 자체가 없음).
    # 그래서 EBIT(영업이익)을 주 지표로 쓰고, 감가상각이 잡힌 종목만 EBITDA를 병행한다.
    # EBITDA가 없는 종목을 추정치로 채우면 그 종목만 조용히 다른 잣대로 재게 된다.
    m["ebit"] = op
    ebitda = (op + da) if (op is not None and da is not None) else None
    m["ebitda"] = ebitda
    m["net_debt_ebit"] = div(net_debt, op)
    m["net_debt_ebitda"] = div(net_debt, ebitda)

    # ---------------- 밸류에이션 ----------------
    # 적자면 None. 이 한 줄이 밸류에이션 랭킹의 신뢰도를 결정한다.
    m["per"] = div(mcap, nip) if (nip or 0) > 0 else None
    # 보통주 귀속 자본 = 지배주주지분 − 신종자본증권.
    # 신종자본증권은 회계상 자본이지만 보통주 몫이 아니다. 금융지주에 특히 크다.
    hybrid = _num(g("hybrid_capital")) or 0.0
    eq_common = (eqp - hybrid) if eqp is not None else None
    m["hybrid_capital"] = hybrid or None
    m["equity_common"] = eq_common
    m["pbr"] = div(mcap, eq_common)
    m["psr"] = div(mcap, rev)
    ev = (mcap + net_debt) if (mcap is not None and net_debt is not None) else None
    m["ev"] = ev
    m["ev_ebit"] = div(ev, op)          # 주 지표 — 전 종목 계산 가능
    m["ev_ebitda"] = div(ev, ebitda)    # 보조 — 감가상각이 잡힌 종목만
    # shares_out 은 **유통주식수**(상장 보통+우선 − 자기주식)를 넣는다. src/shares.py 참조.
    m["eps"] = div(nip, shares_out)
    m["bps"] = div(eq_common, shares_out)
    m["fcf_yield"] = _pct(div(fcf, mcap))

    # ---------------- 성장성 ----------------
    if hist is not None and not hist.empty:
        m.update(_growth(row, hist))
        m.update(_normalized(row, hist, mcap, tax_rate))

    if m.get("per") is not None and m.get("eps_cagr_3y") not in (None, 0):
        cagr_pct = m["eps_cagr_3y"]
        m["peg"] = m["per"] / cagr_pct if cagr_pct and cagr_pct > 0 else None
    else:
        m["peg"] = None

    # ---------------- 통화 불일치 무효화 ----------------
    # 재무제표가 원화가 아니면(두산밥캣 USD, 외국기업 950xxx USD·JPY) 시가총액(원)과
    # 나누는 지표가 전부 무의미해진다. 두산밥캣 PBR이 1,180으로 잡혀 있었다.
    # 환산하려면 시점별 환율이 필요한데 지금 그 데이터가 없다. 틀린 값을 남기는 것보다
    # 비우는 게 낫다 — grading.py가 결측 지표를 빼고 가중치를 재정규화한다.
    # 재무제표 안에서 끝나는 지표(ROE·영업이익률·부채비율·성장률)는 통화와 무관하므로 남긴다.
    # NaN 은 파이썬에서 참이라 `or "KRW"` 폴백이 먹지 않는다. 과거 테이블
    # (fin_ttm_hist)에는 이 컬럼이 없어 concat 시 NaN이 되므로 명시적으로 거른다.
    raw_cur = g("currency")
    cur = "KRW" if raw_cur is None or (isinstance(raw_cur, float) and math.isnan(raw_cur)) \
        else str(raw_cur).strip().upper() or "KRW"
    m["currency"] = cur
    if cur != "KRW":
        for k in MIXED_UNIT_METRICS:
            m[k] = None

    # ---------------- 트랙별 무효화 ----------------
    for k in config.TRACK_NULL_METRICS.get(track, ()):
        m[k] = None

    return m


def _pct(x):
    return None if x is None else x * 100.0


def _lag(hist: pd.DataFrame, period: int, col: str, back: int):
    """`back` 분기 전 값. 결측이면 None."""
    row = hist[hist["period"] == period - back]
    if row.empty or col not in row.columns:
        return None
    return _num(row.iloc[0][col])


def _growth(row: dict, hist: pd.DataFrame) -> dict:
    p = row["period"]
    out: dict[str, float | None] = {}

    pairs = [("revenue_ttm", "revenue_growth_yoy"),
             ("operating_income_ttm", "op_growth_yoy"),
             ("net_income_ttm", "ni_growth_yoy")]
    for col, name in pairs:
        cur, prev = _num(row.get(col)), _lag(hist, p, col, 4)
        # 전년이 적자면 성장률이 의미를 잃는다(부호가 뒤집혀 +수천% 가 나온다).
        out[name] = _pct(cur / prev - 1) if (cur is not None and prev and prev > 0) else None

    cur3, prev3 = _num(row.get("revenue_ttm")), _lag(hist, p, "revenue_ttm", 12)
    out["revenue_cagr_3y"] = (
        _pct((cur3 / prev3) ** (1 / 3) - 1)
        if (cur3 is not None and prev3 and prev3 > 0 and cur3 > 0) else None)

    epsc, epsp = _num(row.get("net_income_parent_ttm")), _lag(hist, p, "net_income_parent_ttm", 12)
    out["eps_cagr_3y"] = (
        _pct((epsc / epsp) ** (1 / 3) - 1)
        if (epsc is not None and epsp and epsp > 0 and epsc > 0) else None)

    # 성장 변동성·가속도 — 최근 8분기 단일분기 매출 YoY
    yoys = []
    for k in range(config.GROWTH_VOL_QUARTERS):
        cur = _lag(hist, p + 1, "revenue_ttm", k + 1) if k else _num(row.get("revenue_ttm"))
        prev = _lag(hist, p, "revenue_ttm", k + 4)
        yoys.append(cur / prev - 1 if (cur is not None and prev and prev > 0) else np.nan)
    arr = np.array(yoys, dtype=float)
    valid = arr[~np.isnan(arr)]
    out["growth_volatility"] = _pct(float(np.std(valid))) if len(valid) >= 4 else None
    if len(valid) >= 4:
        recent, older = arr[:2], arr[2:4]
        if not np.isnan(recent).any() and not np.isnan(older).any():
            out["growth_acceleration"] = _pct(float(recent.mean() - older.mean()))
        else:
            out["growth_acceleration"] = None
    else:
        out["growth_acceleration"] = None
    return out


def _normalized(row: dict, hist: pd.DataFrame, mcap, tax_rate) -> dict:
    """정상화 이익 — 최근 5년 평균 영업이익률 × 현재 매출.

    사이클 산업은 호황 PER이 낮고 불황 PER이 무한대가 된다.
    실적 PER과 정상화 PER의 격차 자체가 지금이 사이클의 어디인지를 알려준다.
    """
    p = row["period"]
    win = hist[(hist["period"] <= p) & (hist["period"] > p - config.NORMALIZE_QUARTERS)]
    if win.empty or "revenue_ttm" not in win or "operating_income_ttm" not in win:
        return {"normalized_income": None, "normalized_per": None}
    margins = (win["operating_income_ttm"] / win["revenue_ttm"]).replace(
        [np.inf, -np.inf], np.nan).dropna()
    rev = _num(row.get("revenue_ttm"))
    if len(margins) < 8 or rev is None:
        return {"normalized_income": None, "normalized_per": None}
    norm_op = float(margins.mean()) * rev
    norm_ni = norm_op * (1 - tax_rate)
    return {
        "normalized_income": norm_ni,
        "normalized_per": div(mcap, norm_ni) if norm_ni > 0 else None,
    }


def winsorize(s: pd.Series, pct: float = config.WINSOR_PCT) -> pd.Series:
    """상하위 pct 클리핑. 원본은 호출부에서 별도 컬럼으로 보존한다."""
    v = s.dropna()
    if len(v) < 20:
        return s
    lo, hi = v.quantile(pct), v.quantile(1 - pct)
    return s.clip(lo, hi)
