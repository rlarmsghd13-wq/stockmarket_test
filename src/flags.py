"""L2 — 적자 이력·구조변경 플래그.

지표 값이 정상 범위여도 그 값을 믿으면 안 되는 상황이 있다.
여기서 그런 상황을 표시해 뒤 단계(등급화·전략)가 알고 쓰게 만든다.
"""
from __future__ import annotations

import pandas as pd

import config

# 적자 이력
TEMP_LOSS = "TEMP_LOSS"          # 1년 — 정상화 이익으로 대체 평가
STRUCT_DOUBT = "STRUCT_DOUBT"    # 2년 — 감점 + 가치 전략 탈락
STRUCT_FAIL = "STRUCT_FAIL"      # 3년 — 유니버스 제외
CRITICAL = "CRITICAL"            # 존속 위험 — 즉시 제외
RESTRUCTURE = "RESTRUCTURE"      # 인적분할·합병 등 — TTM 비교 불가


def loss_streak(annual: pd.DataFrame, col: str = "operating_income") -> int:
    """최근 연속 영업적자 연수. 연간(4분기) 행만 넣는다."""
    if annual.empty or col not in annual.columns:
        return 0
    s = annual.sort_values("year")[col]
    n = 0
    for v in reversed(s.tolist()):
        if pd.isna(v):
            break
        if v < 0:
            n += 1
        else:
            break
    return n


def loss_flag(streak: int, *, track: str, debt_ratio=None, ocf=None) -> str | None:
    """연속 적자 연수 → 플래그. T3 제약·바이오는 적용하지 않는다.

    임상 단계 바이오텍은 적자가 정상 상태다. 3년 룰로 자르면 트랙이 비어버린다.
    대신 `cash_runway_flag`를 쓴다.
    """
    if track == config.TRACK_PHARMA or streak == 0:
        return None
    if (debt_ratio is not None and debt_ratio > config.CRITICAL_DEBT_RATIO
            and ocf is not None and ocf < 0):
        return CRITICAL
    if streak >= config.LOSS_FAIL:
        return STRUCT_FAIL
    if streak >= config.LOSS_DOUBT:
        return STRUCT_DOUBT
    return TEMP_LOSS


def cash_runway_quarters(cash, quarterly_burn) -> float | None:
    """T3 대체 지표 — 돈이 몇 분기 남았는가."""
    if cash is None or quarterly_burn is None or quarterly_burn <= 0:
        return None
    return cash / quarterly_burn


def runway_grade(quarters: float | None) -> str | None:
    if quarters is None:
        return None
    if quarters < config.RUNWAY_DANGER:
        return "D"
    if quarters < config.RUNWAY_WARN:
        return "C"
    return None


def restructure_flags(t: pd.DataFrame, *, drop: float = 0.20,
                      rise: float = 0.50) -> pd.DataFrame:
    """인적분할·합병 탐지 — 자본이나 자산이 한 분기에 급변한 지점.

    왜 필요한가: TTM은 `당기누적 + 전년연간 − 전년동기누적`으로 만든다.
    이 구간에 인적분할이 끼면 **분할 전 회사와 분할 후 회사의 실적이 한 숫자에 섞인다.**
    실제로 삼성바이오로직스는 2025년 4분기에 자본이 12.2조 → 7.5조로 떨어졌고,
    그 여파로 TTM 기준 EPS가 벤더 공표값과 45% 어긋난다. 계산이 틀린 게 아니라
    **서로 다른 회사를 더한 값**이기 때문이다.

    이후 4분기 동안 TTM 기반 성장률·PER은 신뢰할 수 없다고 표시한다.

    상승과 하락에 다른 기준을 쓴다. 자본이 **줄어드는** 것은 분할·소각·손상 같은
    구조적 사건이라야 설명되지만, **느는** 것은 호황기 이익 유보만으로도 한 분기에
    30~40%가 나온다.

    그리고 결정적으로 **자산이 같이 움직였는지**를 본다. 사업이 통째로 떨어져 나가면
    자본과 자산이 같은 방향으로 함께 줄지만, 적자 누적이나 기타포괄손익 변동은
    자본만 움직인다. 실측으로 확인된 판별력:

        삼성바이오로직스 2025Q4  자본 −38.8% / 자산 −39.7%  → 인적분할 (진짜)
        올릭스        2022Q1  자본 −46.1% / 자산  −8.5%  → 적자 누적
        삼성생명       2025Q4  자본 +58.4% / 자산 +10.0%  → 채권평가손익
        한화오션       2022Q3  자본 −42.0% / 자산  +4.0%  → 적자

    자본만 보면 200종목 중 54종목이 걸리지만, 자산 동반 조건을 걸면 실제 구조 사건만 남는다.
    """
    if t.empty or "equity" not in t.columns:
        return pd.DataFrame(columns=["ticker", "period", "flag", "note"])

    d = t.sort_values("period").copy()
    prev_eq = d["equity"].shift(1)
    change = (d["equity"] - prev_eq) / prev_eq.abs()

    if "assets" in d.columns:
        prev_as = d["assets"].shift(1)
        as_change = (d["assets"] - prev_as) / prev_as.abs()
        # 자산이 자본과 같은 방향으로 절반 이상 따라 움직였는가
        together = (as_change * change > 0) & (as_change.abs() > drop * 0.5)
    else:
        together = pd.Series(True, index=d.index)

    # 자본이 **늘어난** 경우, 그 증가분이 이익으로 설명되면 유기적 성장이다.
    # 호황기에는 이익 유보만으로 한 분기 자본이 크게 뛴다 (실측: SK하이닉스
    # 자본 +60% / 자산 +57%). 이걸 구조변경으로 잡으면 실적이 가장 좋은 종목을
    # 골라서 신뢰 불가로 만든다. 자본 감소는 이익으로 설명될 수 없으므로 그대로 둔다.
    delta_eq = d["equity"] - prev_eq
    if "net_income_ttm" in d.columns:
        explained = delta_eq <= d["net_income_ttm"].abs() * 1.2
    else:
        explained = pd.Series(False, index=d.index)

    is_drop = change < -drop
    is_rise = (change > rise) & ~explained.fillna(False)
    hits = d[(is_drop | is_rise) & together & prev_eq.notna()]
    rows = []
    for r in hits.itertuples():
        rows.append({
            "ticker": r.ticker, "period": r.period, "flag": RESTRUCTURE,
            "note": (f"자본 {change.loc[r.Index]*100:+.0f}% · "
                     f"자산 {as_change.loc[r.Index]*100:+.0f}% 동반 변동"),
        })
        # 이 시점부터 4분기 동안 TTM 비교 불가
        for k in range(1, 4):
            rows.append({
                "ticker": r.ticker, "period": r.period + k, "flag": RESTRUCTURE,
                "note": "구조변경 직후 — TTM 성장률·PER 신뢰 불가",
            })
    return pd.DataFrame(rows)


def ttm_reliable(flags_df: pd.DataFrame, ticker: str, period: int) -> bool:
    """이 (종목, 시점)의 TTM 파생 지표를 믿어도 되는가."""
    if flags_df.empty:
        return True
    hit = flags_df[(flags_df["ticker"] == ticker) & (flags_df["period"] == period)
                   & (flags_df["flag"] == RESTRUCTURE)]
    return hit.empty
