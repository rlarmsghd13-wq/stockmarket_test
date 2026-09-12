"""L3 — 등급화. 원시 지표를 부문별 점수와 A+~D 등급으로.

PER 12는 좋은 숫자인가? 은행이면 비싸고 반도체 장비면 싸다. 절대 임계값을 쓰면
특정 업종이 통째로 상위나 하위를 점령한다. 그래서 **트랙 → 업종 2단계 백분위**로
정규화한다. 트랙이 다르면 애초에 같은 표에 올라오지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# 부문 → [(지표, 가중치, 방향)] · 방향 -1 은 "낮을수록 좋음"
CATEGORIES: dict[str, list[tuple[str, float, int]]] = {
    # 3년 매출 CAGR을 뺐다 — 검증에서 IC -0.029로 **역방향**이었다.
    # 과거 3년 성장이 높을수록 이후 3개월 수익률이 낮았다. 빠진 0.20은 재배분.
    "growth": [
        ("revenue_growth_yoy", 0.35, +1),
        ("op_growth_yoy", 0.35, +1),
        ("ni_growth_yoy", 0.20, +1),
        ("growth_volatility", 0.10, -1),      # 들쭉날쭉한 성장은 성장이 아니다
    ],
    "profitability": [
        ("roe", 0.35, +1),
        ("operating_margin", 0.25, +1),
        ("roic", 0.20, +1),
        ("roa", 0.10, +1),
        ("net_margin", 0.10, +1),
    ],
    "stability": [
        ("debt_ratio", 0.30, -1),
        ("current_ratio", 0.20, +1),
        ("ocf_to_ni", 0.25, +1),              # 이익의 질
        ("fcf_to_ni", 0.25, +1),
    ],
    "valuation": [
        ("per", 0.30, -1),
        ("pbr", 0.25, -1),
        ("ev_ebit", 0.20, -1),
        ("psr", 0.10, -1),
        ("fcf_yield", 0.15, +1),
    ],
    # 월 단위로 실제로 바뀌는 유일한 부문. 이격도는 과열 지표라 역방향 —
    # 높을수록 추격매수 위험이다.
    #
    # 투자자별 순매수(외국인·기관)는 제외했다. KRX가 이 엔드포인트에서 긴 구간을
    # 거부해 4년 창에서 200종목 중 65종목만 받아졌고, 결측이 큰 지표를 넣으면
    # 종목마다 다른 잣대로 재게 된다. 가중치는 남은 다섯 지표로 재배분했다.
    "momentum": [
        ("return_3m", 0.35, +1),
        ("return_1m", 0.25, +1),
        ("volume_ratio", 0.20, +1),
        ("return_6m", 0.13, +1),
        ("disparity_20", 0.07, -1),
    ],
}

# T2 금융은 지표셋이 다르다. 없는 지표는 자동으로 빠지고 가중치가 재정규화된다.
TRACK_OVERRIDE: dict[str, dict[str, list[tuple[str, float, int]]]] = {
    config.TRACK_FINANCE: {
        "profitability": [("roa", 0.45, +1), ("roe", 0.35, +1), ("net_margin", 0.20, +1)],
        "stability": [("equity_ratio", 1.00, +1)],   # 자기자본비율 (자본/자산)
        "valuation": [("pbr", 0.50, -1), ("per", 0.30, -1), ("fcf_yield", 0.20, +1)],
    },
    config.TRACK_PHARMA: {
        "profitability": [("gp_to_assets", 0.60, +1), ("roa", 0.40, +1)],
        "valuation": [("psr", 0.60, -1), ("pbr", 0.40, -1)],
    },
    config.TRACK_HOLDING: {
        "valuation": [("pbr", 0.60, -1), ("ev_ebit", 0.40, -1)],
    },
}

MIN_PEERS = 8          # 이보다 적으면 트랙 전체로 폴백

# T2 하위업종은 자본구조가 근본적으로 다르다. 실측 자기자본비율:
#   은행·금융지주 KB 7.2% · 신한 7.4% · 하나 6.8%
#   보험         삼성생명 31.9% · 삼성화재 31.4%
# 4배 차이라 한 표본에 넣으면 보험사가 안정성 상위를 자동으로 독점한다.
# T1/T2를 나눈 것과 같은 문제가 T2 안에서 반복되므로, 표본이 작아도 쪼갠다.
# (n=5면 백분위가 거칠지만, 다른 자로 잰 값을 섞는 것보다 낫다)
MIN_PEERS_BY_TRACK = {config.TRACK_FINANCE: 4}

GRADE_CUTS = [(90, "A+"), (75, "A"), (45, "B"), (20, "C"), (0, "D")]


def grade(pct: float | None) -> str | None:
    if pct is None or pd.isna(pct):
        return None
    for cut, g in GRADE_CUTS:
        if pct >= cut:
            return g
    return "D"


def _spec(track: str) -> dict[str, list[tuple[str, float, int]]]:
    base = {k: list(v) for k, v in CATEGORIES.items()}
    base.update(TRACK_OVERRIDE.get(track, {}))
    return base


def _percentile(s: pd.Series, direction: int) -> pd.Series:
    """표본 내 백분위 0~100. 결측은 결측으로 남긴다 — 중간값으로 채우지 않는다."""
    v = s.astype(float)
    if v.notna().sum() < 3:
        return pd.Series(np.nan, index=s.index)
    p = v.rank(pct=True, na_option="keep") * 100
    return p if direction > 0 else 100 - p


def build(metrics: pd.DataFrame, sector_map: dict[str, str]) -> pd.DataFrame:
    """지표 테이블 → 부문 점수·등급.

    sector_map: {ticker: KSIC 앞 2자리}
    """
    df = metrics.copy()
    df["sector"] = df["ticker"].map(sector_map).fillna("00")

    out_rows = []
    for track, tg in df.groupby("track"):
        spec = _spec(str(track))
        # 업종 표본이 기준 미만이면 트랙 전체로 폴백한다.
        min_peers = MIN_PEERS_BY_TRACK.get(str(track), MIN_PEERS)
        counts = tg["sector"].value_counts()
        tg = tg.assign(peer=np.where(
            tg["sector"].map(counts).fillna(0) >= min_peers,
            tg["sector"], "__TRACK__"))

        for cat, members in spec.items():
            score = pd.Series(0.0, index=tg.index)
            wsum = pd.Series(0.0, index=tg.index)
            for col, w, direction in members:
                if col not in tg.columns:
                    continue
                pct = tg.groupby("peer")[col].transform(
                    lambda s, d=direction: _percentile(s, d))
                ok = pct.notna()
                score = score.add(pct.fillna(0) * w * ok, fill_value=0)
                wsum = wsum.add(w * ok, fill_value=0)
            # 가중치 재정규화 — 결측 지표는 빠지고 남은 것끼리 100%가 된다
            tg[f"{cat}_score"] = np.where(wsum > 0, score / wsum, np.nan)
            tg[f"{cat}_grade"] = [grade(x) for x in tg[f"{cat}_score"]]
            tg[f"{cat}_n"] = wsum.round(2)
        out_rows.append(tg)

    res = pd.concat(out_rows, ignore_index=True)
    cats = list(CATEGORIES)
    res["quality_score"] = res[[f"{c}_score" for c in
                                ("profitability", "stability")]].mean(axis=1)
    res["coverage"] = res[[f"{c}_n" for c in cats]].notna().sum(axis=1)
    return res
