"""적립식 매매 시뮬레이터 — 매달 정액 납입 + 손절·분할익절.

17번 백테스트와 다른 점
    17번은 "매달 상위 5개로 갈아타기"였다. 순위에서 밀리면 곧바로 팔았다.
    여기서는 **순위에서 밀려도 팔지 않는다.** 손절이나 익절에 닿을 때만 판다.
    그래서 시간이 지날수록 보유 종목이 쌓이고, 현금흐름(월 납입)이 성과에 섞인다.

규칙
    · 매달 기준일에 CONTRIB원을 넣고, 그날 상위 N종목에 **현금을 균등 분배**
      (이미 들고 있는 종목이면 추가 매수 — 평단이 바뀐다)
    · 손절: 보유분 평단 대비 -STOP% 종가 도달 시 전량
    · 1차 익절: +T1% 도달 시 **절반**
    · 2차 익절: +T2% 도달 시 **잔량**
    · 매도 대금은 현금으로 두었다가 다음 달 매수일에 함께 분배

측정
    납입 시점이 제각각이라 단순 수익률로는 비교할 수 없다. 그래서 펀드처럼
    **좌수(unit)**를 써서 기준가를 만든다 — 납입은 좌수를 늘릴 뿐 기준가를
    움직이지 않으므로, 기준가 수익률(TWR)은 전략 자체의 성적이 된다.
    실제 체감은 **총 납입액 대비 최종 평가액**으로 따로 본다.

한계
    · 손절·익절 판정은 **종가** 기준이다. 장중 고가·저가는 수정주가가 아니라
      섞어 쓸 수 없다. 실제로는 장중에 닿아 더 일찍 체결될 수 있다.
    · 증권사 목표가는 과거 데이터가 없어 **매수가 대비 비율**로 대체한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COST = 0.35          # 편도 %  (수수료 0.015×2 + 거래세 0.18 + 슬리피지 0.15)
CONTRIB = 500_000    # 월 납입액
N_TOP = 5


@dataclass
class Lot:
    ticker: str
    qty: float                 # 주식 수 (소수 허용 — 금액 기반 시뮬레이션)
    cost_basis: float          # 평단
    half_done: bool = False
    first_day: object = None   # 처음 편입한 날 — 보유기간 측정용


@dataclass
class Result:
    nav: pd.Series                      # 기준가 (좌수 기준)
    value: pd.Series                    # 평가액 + 현금
    contributed: float
    final: float
    trades: pd.DataFrame                # 청산 기록
    holdings: pd.Series                 # 일별 보유 종목 수
    cash_ratio: pd.Series
    buys: pd.DataFrame                  # 매수 기록 (회전율 측정용)


def _pivot(px: pd.DataFrame) -> pd.DataFrame:
    p = px.pivot_table(index="date", columns="ticker", values="close_adj", aggfunc="last")
    return p.sort_index()


def simulate(picks: dict[str, list[str]], px: pd.DataFrame, asofs: list[str], *,
             contrib: float = CONTRIB, n_top: int = N_TOP, initial: float = 0.0,
             alloc: str = "equal",
             stop: float | None = None, t1: float | None = None,
             t2: float | None = None, cost: float = COST) -> Result:
    """picks: {기준일: [상위 종목코드...]} · asofs: 전체 기준일 목록.

    **납입은 매달 빠짐없이 한다.** 기준일이 휴장일이면 그 다음 거래일에 넣고,
    고를 종목이 없는 달(게이트 통과 0개)은 현금으로 들고 간다. 이걸 빼먹으면
    전략마다 납입액이 달라져 비교가 성립하지 않는다.

    initial: 첫 매수일에 넣는 초기 자본
    alloc  : 가용 현금을 상위 종목에 어떻게 배분하는가
      equal     — 현금을 상위 n종목에 1/n씩 (이미 보유 중이면 추가 매수)
      fill      — 목표비중(1/n)에 모자란 종목부터 채운다. 팔지는 않는다
      rebalance — 매달 전체를 상위 n종목 균등으로 맞춘다. **순위에서 밀리면 매도**
    """
    prices = _pivot(px)
    idx = prices.index
    buy_set: dict[pd.Timestamp, list[str]] = {}
    for a in sorted(asofs):
        pos = idx.searchsorted(pd.Timestamp(a), side="left")
        if pos < len(idx):
            buy_set[idx[pos]] = picks.get(a, [])
    if not buy_set:
        raise ValueError("매수일이 없습니다")
    # 마지막 기준일에서 끊는다. 그 뒤 주가까지 받으면 같은 기간으로 끊긴
    # 벤치마크보다 유리해져 비교가 어긋난다(전체 구간에서 3개월 차이가 났다).
    days = idx[(idx >= min(buy_set)) & (idx <= max(buy_set))]

    lots: dict[str, Lot] = {}
    cash = 0.0
    units = 0.0
    nav0 = 1000.0
    contributed = 0.0
    navs, vals, hold_n, cash_r, trades, buys = [], [], [], [], [], []
    last_px: dict[str, float] = {}

    def value_of(row) -> float:
        v = cash
        for t, lot in lots.items():
            p = row.get(t)
            if p is None or np.isnan(p):
                p = last_px.get(t, lot.cost_basis)
            v += lot.qty * p
        return v

    def sell(t: str, frac: float, price: float, reason: str, day) -> None:
        nonlocal cash
        lot = lots[t]
        q = lot.qty * frac
        gross = q * price
        cash += gross * (1 - cost / 100)
        trades.append({"date": day, "ticker": t, "reason": reason,
                       "ret_pct": (price / lot.cost_basis - 1) * 100,
                       "amount": gross,
                       "hold_days": (day - lot.first_day).days if lot.first_day else None})
        lot.qty -= q
        if lot.qty <= 1e-9:
            del lots[t]

    for day in days:
        row = prices.loc[day]
        for t, p in row.dropna().items():
            last_px[t] = float(p)

        # ── 매도 판정 (종가 기준, 손절 우선) ─────────────────────────
        for t in list(lots):
            lot = lots[t]
            p = row.get(t)
            if p is None or np.isnan(p):
                continue
            p = float(p)
            chg = p / lot.cost_basis - 1
            if stop is not None and chg <= -stop / 100:
                sell(t, 1.0, p, "손절", day)
            elif t2 is not None and lot.half_done and chg >= t2 / 100:
                sell(t, 1.0, p, "2차 익절", day)
            elif t1 is not None and not lot.half_done and chg >= t1 / 100:
                sell(t, 0.5, p, "1차 익절", day)
                if t in lots:
                    lots[t].half_done = True

        # ── 상장폐지·거래정지 정리 (60거래일 넘게 시세 없음) ─────────
        for t in list(lots):
            s = prices[t]
            fut = s.loc[day:].dropna()
            if fut.empty:
                sell(t, 1.0, last_px.get(t, lots[t].cost_basis), "상장폐지", day)

        # ── 매수일: 납입 + 배분 ─────────────────────────────────────
        if day in buy_set:
            add = contrib + (initial if contributed == 0 else 0.0)
            v_before = value_of(row)
            if units == 0:
                units = add / nav0
            else:
                units += add / (v_before / units)
            cash += add
            contributed += add

            targets = [t for t in buy_set[day][:n_top]
                       if t in row.index and not np.isnan(row.get(t, np.nan))]

            def buy(t: str, amount: float) -> None:
                nonlocal cash
                if amount <= 0:
                    return
                buys.append({"date": day, "ticker": t, "amount": amount})
                p = float(row[t])
                q = amount * (1 - cost / 100) / p
                if t in lots:
                    lot = lots[t]
                    tot = lot.qty + q
                    lot.cost_basis = (lot.cost_basis * lot.qty + p * q) / tot
                    lot.qty = tot
                    lot.half_done = False     # 평단이 바뀌면 익절 단계 초기화
                else:
                    lots[t] = Lot(t, q, p, first_day=day)
                cash -= amount

            if targets:
                if alloc == "rebalance":
                    # 상위에서 밀린 보유는 전량 매도 → 전체를 균등으로
                    for t in list(lots):
                        if t not in targets:
                            p = row.get(t)
                            sell(t, 1.0, float(p) if p is not None and not np.isnan(p)
                                 else last_px.get(t, lots[t].cost_basis), "순위이탈", day)
                    total = value_of(row)
                    tgt = total / len(targets)
                    for t in targets:                      # 초과분 먼저 매도
                        cur = lots[t].qty * float(row[t]) if t in lots else 0.0
                        if cur > tgt * 1.01:
                            sell(t, (cur - tgt) / cur, float(row[t]), "리밸런싱", day)
                    for t in targets:                      # 부족분 매수
                        cur = lots[t].qty * float(row[t]) if t in lots else 0.0
                        buy(t, min(cash, max(0.0, tgt - cur)))
                elif alloc == "fill":
                    total = value_of(row)
                    tgt = total / len(targets)
                    need = {t: max(0.0, tgt - (lots[t].qty * float(row[t]) if t in lots else 0.0))
                            for t in targets}
                    s_need = sum(need.values())
                    if s_need > 0:
                        for t in targets:
                            buy(t, min(cash, cash * need[t] / s_need) if s_need > cash
                                else need[t])
                    if cash > 1:                           # 남으면 균등 분배
                        per = cash / len(targets)
                        for t in targets:
                            buy(t, per)
                else:                                      # equal
                    per = cash / len(targets)
                    for t in targets:
                        buy(t, per)
                cash = max(cash, 0.0)

        v = value_of(row)
        navs.append(v / units if units else nav0)
        vals.append(v)
        hold_n.append(len(lots))
        cash_r.append(cash / v * 100 if v else 0.0)

    idx = pd.Index(days, name="date")
    return Result(nav=pd.Series(navs, index=idx), value=pd.Series(vals, index=idx),
                  contributed=contributed, final=float(vals[-1]),
                  trades=pd.DataFrame(trades), holdings=pd.Series(hold_n, index=idx),
                  cash_ratio=pd.Series(cash_r, index=idx), buys=pd.DataFrame(buys))


def mdd(s: pd.Series) -> float:
    return float((s / s.cummax() - 1).min() * 100)


def irr_monthly(contrib: float, n: int, final: float) -> float:
    """월 정액 납입 n회 → 최종 final 이 되는 월수익률(연율 %)."""
    lo, hi = -0.9, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        fv = sum(contrib * (1 + mid) ** (n - i) for i in range(n))
        if fv > final:
            hi = mid
        else:
            lo = mid
    r = (lo + hi) / 2
    return ((1 + r) ** 12 - 1) * 100


def summarize(res: Result, label: str) -> dict:
    yrs = len(res.nav) / 252
    twr = res.nav.iloc[-1] / res.nav.iloc[0] - 1
    n_month = int(round(res.contributed / CONTRIB))
    tr = res.trades
    wins = tr[tr["ret_pct"] > 0] if len(tr) else tr
    return {
        "전략": label,
        "납입": res.contributed,
        "최종": res.final,
        "납입대비": (res.final / res.contributed - 1) * 100 if res.contributed else np.nan,
        "기준가수익": twr * 100,
        "CAGR": ((1 + twr) ** (1 / yrs) - 1) * 100,
        "MDD": mdd(res.nav),
        "IRR": irr_monthly(CONTRIB, n_month, res.final),
        "매도건수": len(tr),
        "승률": (len(wins) / len(tr) * 100) if len(tr) else np.nan,
        "평균보유": res.holdings.mean(),
        "최대보유": int(res.holdings.max()),
        "현금비중": res.cash_ratio.mean(),
    }
