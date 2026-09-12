"""L6 — 종목 분석 리포트 데이터 조립.

**모든 문장에 계산된 수치가 붙는다.** 근거 불릿은 규칙으로 생성하며,
데이터에 없는 내용은 만들어내지 않는다. 지금 단계에서 채울 수 없는 항목
(증권사 목표주가, 공시·IR 기반 정성 근거)은 비워두고 그렇다고 표시한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from . import momentum, regime, strategies

CATS = ["growth", "profitability", "stability", "valuation", "momentum"]
CAT_KR = {"growth": "성장성", "profitability": "수익성", "stability": "안정성",
          "valuation": "밸류에이션", "momentum": "모멘텀·수급"}

# 근거 불릿에 쓸 지표 — (컬럼, 표시명, 포맷, 높을수록 좋은가)
FACTS = [
    ("roe", "ROE", "{:.1f}%", True),
    ("operating_margin", "영업이익률", "{:.1f}%", True),
    ("roic", "ROIC", "{:.1f}%", True),
    ("revenue_growth_yoy", "매출성장률(YoY)", "{:+.1f}%", True),
    ("op_growth_yoy", "영업이익성장률(YoY)", "{:+.1f}%", True),
    ("revenue_cagr_3y", "매출 3년 CAGR", "{:+.1f}%", True),
    ("ocf_to_ni", "영업CF/순이익", "{:.0f}%", True),
    ("fcf_to_ni", "FCF/순이익", "{:.0f}%", True),
    ("debt_ratio", "부채비율", "{:.0f}%", False),
    ("current_ratio", "유동비율", "{:.0f}%", True),
    ("per", "PER", "{:.1f}배", False),
    ("pbr", "PBR", "{:.2f}배", False),
    ("ev_ebit", "EV/EBIT", "{:.1f}배", False),
    ("fcf_yield", "FCF수익률", "{:.1f}%", True),
    ("return_3m", "3개월 수익률", "{:+.1f}%", True),
]


def _fmt(v, f):
    return "—" if v is None or pd.isna(v) else f.format(v)


def band(prices: pd.DataFrame, ttm: pd.DataFrame, ticker: str,
         shares: float, asof: str) -> dict:
    """자기 자신의 과거 밸류에이션 밴드에서 지금 어디인가.

    "얼마가 싼가"의 기준을 시장 평균이 아니라 그 종목의 과거에서 가져온다.
    수집한 시세 창(2년)만큼만 산출되므로 설계 초안의 5년보다 짧다.
    """
    out = {"per_p25": None, "per_p50": None, "per_p75": None,
           "per_pos": None, "pbr_pos": None, "days": 0}
    p = prices[prices["ticker"] == ticker].copy()
    h = ttm[ttm["ticker"] == ticker].copy()
    if p.empty or h.empty or not shares:
        return out
    p["date"] = pd.to_datetime(p["date"])
    p = p[p["date"] <= pd.Timestamp(asof)].sort_values("date")
    h["rd"] = pd.to_datetime(h["rcept_dt"])
    h = h.sort_values("rd")
    keep = ["rd", "net_income_parent_ttm"] + [
        c for c in ("equity_parent", "hybrid_capital") if c in h.columns]
    m = pd.merge_asof(p[["date", "close_adj"]], h[keep],
                      left_on="date", right_on="rd", direction="backward")
    eqc = m["equity_parent"] - (m["hybrid_capital"].fillna(0)
                                if "hybrid_capital" in m else 0)
    per = (m["close_adj"] * shares / m["net_income_parent_ttm"]).replace(
        [np.inf, -np.inf], np.nan).dropna()
    pbr = (m["close_adj"] * shares / eqc).replace(
        [np.inf, -np.inf], np.nan).dropna()
    out["days"] = len(m)
    if len(per) > 60:
        cur = per.iloc[-1]
        out.update(per_p25=float(per.quantile(.25)), per_p50=float(per.quantile(.50)),
                   per_p75=float(per.quantile(.75)),
                   per_pos=float((per < cur).mean() * 100))
    if len(pbr) > 60:
        out["pbr_pos"] = float((pbr < pbr.iloc[-1]).mean() * 100)
    return out


def verdict(score, gate_pass: bool, held: bool, *, strategy: str = "value",
            row: dict | None = None, prev_row: dict | None = None,
            bd: dict | None = None, defer_months: int = 0) -> dict:
    """매수·매도 판정.

    매수는 **점수만으로** 정한다. 가격 타이밍을 잡지 않는다.
    매도는 점수가 기준 아래로 내려갔을 때인데, **그 하락이 시장 탓이면 보류한다**
    (장기 전략에 한해). 판별은 `regime.defer_sell`이 폭(breadth)으로 한다.
    """
    if score is None or pd.isna(score):
        return {"label": "판정 불가", "code": "na", "why": "점수 산출 불가", "defer": None}
    s = float(score)

    # 매도·축소 국면에서만 보류를 따진다
    def _maybe_defer(base_label: str, base_code: str, why: str) -> dict:
        if not (held and regime.applies_to(strategy) and row is not None
                and bd is not None):
            return {"label": base_label, "code": base_code, "why": why, "defer": None}
        d = regime.defer_sell(row, prev_row, bd, defer_months)
        if d["defer"]:
            return {"label": "보유 (매도 보류)", "code": "defer",
                    "why": f"{why} — 다만 {d['reason']}",
                    "defer": d}
        return {"label": base_label, "code": base_code,
                "why": f"{why} · {d['note']}", "defer": d}

    if not gate_pass:
        if held and regime.applies_to(strategy) and row is not None and bd is not None:
            return _maybe_defer("매도", "sell", "게이트 미통과")
        return {"label": "제외", "code": "out",
                "why": "게이트 미통과 — 점수와 무관하게 후보에서 뺀다", "defer": None}
    if s >= config.SCORE_BUY:
        return {"label": "매수", "code": "buy", "defer": None,
                "why": f"{s:.1f}점 ≥ 신규 매수 기준 {config.SCORE_BUY:.0f}점"}
    if s >= config.SCORE_HOLD:
        if held:
            return {"label": "보유", "code": "hold", "defer": None,
                    "why": f"{s:.1f}점 — 매수 기준({config.SCORE_BUY:.0f}) 미만이나 "
                           f"보유 기준({config.SCORE_HOLD:.0f}) 이상이라 유지"}
        return {"label": "관망", "code": "watch", "defer": None,
                "why": f"{s:.1f}점 — 신규 매수 기준 {config.SCORE_BUY:.0f}점에 미달"}
    if s >= config.SCORE_TRIM:
        if held:
            return _maybe_defer("비중 축소", "trim",
                                f"{s:.1f}점 — 보유 기준 {config.SCORE_HOLD:.0f}점 아래")
        return {"label": "관망", "code": "watch", "defer": None,
                "why": f"{s:.1f}점 — 신규 매수 기준 미달"}
    if held:
        return _maybe_defer("매도", "sell",
                            f"{s:.1f}점 — 매도 기준 {config.SCORE_TRIM:.0f}점 아래")
    return {"label": "제외", "code": "out", "defer": None,
            "why": f"{s:.1f}점 — 매도 기준 {config.SCORE_TRIM:.0f}점 아래"}


def exit_plan(row, bnd: dict) -> list[dict]:
    """매도 계획 — 기본·상방·하방 세 갈래.

    손절 기준은 가격이 아니라 **논거 훼손**이다. 가격 하락은 재점검을 강제하는
    보조 장치일 뿐이다.
    """
    px = row.get("price_asof")
    per = row.get("per")
    out = []
    if px is not None and pd.notna(px) and per and bnd.get("per_p75") and per > 0:
        out.append({
            "case": "기본", "trigger": f"{px * bnd['per_p75'] / per:,.0f}원",
            "action": "1/3 익절",
            "basis": f"2년 PER 밴드 75백분위 ({bnd['per_p75']:.1f}배) 도달"})
    out.append({
        "case": "상방", "trigger": "실적이 가정을 상회",
        "action": "목표 재산정 후 계속 보유",
        "basis": "좋아져서 비싸진 것과 안 변했는데 비싸진 것은 다르다"})
    out.append({
        "case": "하방", "trigger": "논거 훼손 조건 발생",
        "action": "가격과 무관하게 청산",
        "basis": "게이트 이탈 · 감점 트리거 · 치명 플래그 중 하나라도"})
    out.append({
        "case": "점수", "trigger": f"종합점수 {config.SCORE_TRIM:.0f}점 미만",
        "action": "매도",
        "basis": f"보유 기준 {config.SCORE_HOLD:.0f}점 아래로 내려가면 먼저 비중 축소"})
    out.append({
        "case": "보류", "trigger": "같은 트랙 40% 또는 업종 50%가 동반 악화",
        "action": "매도 보류 · 다음 달 재평가",
        "basis": f"시장 요인으로 판단. 누적 {regime.MAX_DEFER_MONTHS}개월까지만"})
    return out


def _cat_of(col: str) -> str | None:
    from .grading import CATEGORIES
    for c, members in CATEGORIES.items():
        if col in [m[0] for m in members]:
            return c
    return None


def evidence(row, gate_row) -> tuple[list[str], list[str]]:
    """찬성·반대 근거.

    **반대 근거를 찬성과 같은 개수로 강제한다.** 임계값만으로 뽑으면
    전 부문이 튼튼한 종목은 반대 근거가 0개가 되고, 그러면 리포트가
    확증편향 기계가 된다. 절대적으로 나쁜 게 없어도 **상대적으로 가장 약한 곳**은
    반드시 있으므로, 부족하면 하위 항목을 순서대로 채운다.
    """
    facts = []
    for col, name, f, _ in FACTS:
        v = row.get(col)
        if v is None or pd.isna(v):
            continue
        cat = _cat_of(col)
        s = row.get(f"{cat}_score") if cat else None
        if s is None or pd.isna(s):
            continue
        facts.append((float(s), f"{name} {_fmt(v, f)} — {CAT_KR.get(cat, cat)} 부문 {s:.0f}점"))

    scored = [(c, row.get(f"{c}_score")) for c in CATS]
    scored = [(c, float(s)) for c, s in scored if s is not None and pd.notna(s)]

    pros = [t for s, t in sorted(facts, key=lambda x: -x[0]) if s >= 70][:3]
    for c, s in sorted(scored, key=lambda x: -x[1])[:2]:
        if s >= 70:
            pros.append(f"{CAT_KR[c]} {s:.0f}점({row.get(c + '_grade')}) — 동종업종 상위 {100-s:.0f}%")
    pros = pros[:4] or [t for _, t in sorted(facts, key=lambda x: -x[0])[:3]]

    # 무조건 들어가야 하는 리스크
    cons = []
    if gate_row is not None and gate_row.get("penalty_why"):
        cons += [f"감점: {w}" for w in str(gate_row["penalty_why"]).split(" · ") if w]
    if row.get("ttm_reliable") is False:
        cons.append("구조변경 직후 — TTM 기반 성장률·PER 신뢰 불가")
    if row.get("loss_streak", 0) > 0:
        cons.append(f"영업적자 {int(row['loss_streak'])}년 연속 ({row.get('loss_flag')})")
    if row.get("stability_distorted"):
        cons.append("금융 부문 연결로 안정성 지표 왜곡 — 세그먼트 확인 필요")
    w = momentum.overheated(row)
    if w:
        cons.append("과열: " + ", ".join(w))
    cov = gate_row.get("coverage") if gate_row else None
    if cov is not None and cov < 4:
        cons.append(f"부문 커버리지 {cov}/4 — 결측 지표가 많아 점수 신뢰도 낮음")

    # 상대적 약점으로 채운다
    ORD = ["가장 낮음", "두 번째로 낮음", "세 번째로 낮음", "네 번째로 낮음"]
    n_cat = len(scored)
    for i, (c, s) in enumerate(sorted(scored, key=lambda x: x[1])):
        if len(cons) >= max(3, len(pros)):
            break
        pos = "하위" if s < 50 else "상위"
        v = s if s < 50 else 100 - s
        rank = ORD[i] if i < len(ORD) else f"{i+1}번째로 낮음"
        cons.append(f"{CAT_KR[c]} {s:.0f}점({row.get(c + '_grade')}) — "
                    f"{n_cat}개 부문 중 {rank}, 동종업종 {pos} {v:.0f}%")
    for s, t in sorted(facts, key=lambda x: x[0]):
        if len(cons) >= max(3, len(pros)):
            break
        if t not in pros:
            cons.append(t)

    n = max(3, min(len(pros), 4))
    return pros[:n], cons[:max(3, n)]


def thesis_breakers(row, strategy: str) -> list[str]:
    """논거 훼손 조건 — 이게 발생하면 가격과 무관하게 청산.

    게이트를 역으로 읽어 "지금 통과 중인 조건이 깨지는 지점"을 적는다.
    손절선이 가격이 아니라 논거인 이유가 여기 있다.
    """
    out = []
    if strategy == strategies.VALUE:
        roe = row.get("roe")
        if pd.notna(roe):
            out.append(f"ROE 8% 미달 (현재 {roe:.1f}%)")
        dr = row.get("debt_ratio")
        if pd.notna(dr):
            out.append(f"부채비율 200% 초과 (현재 {dr:.0f}%)")
        out.append("연간 FCF 적자 2년 연속")
        out.append("영업CF/순이익 50% 미만 2년 연속 (이익의 질 훼손)")
    elif strategy == strategies.GROWTH:
        c = row.get("revenue_cagr_3y")
        if pd.notna(c):
            out.append(f"3년 매출 CAGR 15% 미달 (현재 {c:+.1f}%)")
        out.append("최근 4분기 중 매출 성장 2회 이하")
        out.append("매출 증가 중 영업이익률 3분기 연속 하락")
    else:
        out.append("20일 이격도 130% 초과 (과열)")
        out.append("거래량 급감과 함께 1개월 수익률 마이너스 전환")
        out.append("영업적자 전환")
    return out


def build(ticker: str, grades: pd.DataFrame, scores: pd.DataFrame,
          prices: pd.DataFrame, ttm: pd.DataFrame, strategy: str,
          asof: str, held: bool = False, *, prev_row: dict | None = None,
          bd: dict | None = None, defer_months: int = 0) -> dict:
    g = grades[grades["ticker"] == ticker]
    if g.empty:
        return {}
    row = g.iloc[0].to_dict()
    sc = scores[(scores["ticker"] == ticker) & (scores["strategy"] == strategy)]
    gate_row = sc.iloc[0].to_dict() if not sc.empty else None

    bnd = band(prices, ttm, ticker, row.get("shares_outstanding"), asof)
    pros, cons = evidence(row, gate_row)
    vd = verdict(gate_row.get("score") if gate_row else None,
                 bool(gate_row.get("gate_pass")) if gate_row else False, held,
                 strategy=strategy, row=row, prev_row=prev_row, bd=bd,
                 defer_months=defer_months)

    contrib = []
    for cat, w in strategies.WEIGHTS[strategy].items():
        if not w:
            continue
        s = row.get(f"{cat}_score")
        contrib.append({"cat": CAT_KR[cat], "weight": w,
                        "score": None if s is None or pd.isna(s) else float(s),
                        "points": None if s is None or pd.isna(s) else float(s) * w / 100})

    return {
        "ticker": ticker, "name": row.get("name"), "market": row.get("market"),
        "track": row.get("track"), "asof": asof, "strategy": strategy,
        "strategy_kr": strategies.LABEL[strategy],
        "price": row.get("price_asof"), "market_cap": row.get("market_cap"),
        "fin_period": f"{int(row['fin_year'])}Q{int(row['fin_quarter'])}",
        "rcept_dt": row.get("rcept_dt"),
        "score": gate_row.get("score") if gate_row else None,
        "base_score": gate_row.get("base_score") if gate_row else None,
        "penalty": gate_row.get("penalty") if gate_row else None,
        "gate_pass": gate_row.get("gate_pass") if gate_row else None,
        "gate_fails": gate_row.get("gate_fails") if gate_row else None,
        "coverage": gate_row.get("coverage") if gate_row else None,
        "categories": [{"key": c, "name": CAT_KR[c],
                        "score": None if pd.isna(row.get(f"{c}_score")) else float(row[f"{c}_score"]),
                        "grade": row.get(f"{c}_grade")} for c in CATS],
        "contrib": contrib,
        "metrics": {col: (None if row.get(col) is None or pd.isna(row.get(col))
                          else float(row[col]))
                    for col, *_ in FACTS},
        "band": bnd,
        "verdict": vd,
        "held": held,
        "exit_plan": exit_plan(row, bnd),
        "pros": pros, "cons": cons,
        "breakers": thesis_breakers(row, strategy),
        "flags": {"ttm_reliable": bool(row.get("ttm_reliable", True)),
                  "loss_streak": int(row.get("loss_streak", 0) or 0),
                  "stability_distorted": bool(row.get("stability_distorted", False)),
                  "treasury_ratio": float(row.get("treasury_ratio", 0) or 0)},
    }
