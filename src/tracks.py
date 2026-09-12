"""L0 — 트랙 판정.

회계가 다르면 같은 자로 재지 않는다. 은행의 부채비율 1,200%는 정상이고
임상 단계 바이오텍의 적자도 정상이다. 하나의 지표셋으로 채점하면 이런 기업은
전부 D를 받거나, 반대로 PBR 0.4에 걸려 가치 전략 상위를 통째로 점령한다.

판정 근거는 두 층이다.
  1차 — DART 표준산업분류코드(induty_code)
  2차 — 재무 특성으로 덮어쓴다. 업종코드가 실제 사업 구조를 못 따라가는 경우가 있다.

가장 중요한 2차 규칙은 **지주회사(KSIC 64992) 안에서 금융지주와 일반지주를 가르는 것**이다.
둘이 같은 코드를 쓰는데 재무 구조는 완전히 다르다 (실측: KB금융 부채비율 1,212%,
하나금융지주 1,378% vs SK 149%, LG 12%).
"""
from __future__ import annotations

import re

import pandas as pd

import config

# KSIC 앞 2자리 → 트랙
KSIC_FINANCE = ("64", "65", "66")   # 금융·보험·금융지원서비스
KSIC_PHARMA = ("21",)               # 의료용 물질 및 의약품
KSIC_HOLDING = "64992"              # 지주회사 (금융지주·일반지주 공용)

# 금융지주와 일반지주를 가르는 연결 부채비율. 사이에 걸치는 기업은 거의 없다.
HOLDING_FINANCE_DEBT_RATIO = 500.0

# 금융지주는 이름에도 드러난다. 부채비율을 못 구했을 때의 폴백.
_FINANCE_NAME = re.compile(r"금융지주|은행|생명|화재|해상|손해보험|증권|캐피탈|카드")

# 연결 재무상태표에 이게 크게 잡히면 금융 부문을 연결한 제조사다.
_FINANCE_ASSET = re.compile(r"금융업채권|할부금융|운용리스자산|리스채권|금융리스")
FINANCE_SEGMENT_MIN_RATIO = 0.10    # 총자산 대비

# 금융성 자산이 있다는 것과 그 때문에 안정성 지표가 망가졌다는 것은 다르다.
# 코웨이는 렌탈 사업이라 운용리스자산이 총자산의 67%지만 부채비율은 94%로 멀쩡하다.
# 여기서 지표를 NULL로 만들면 유효한 값을 잃는다. 부채비율까지 실제로
# 부풀어 있을 때만 왜곡으로 본다 (현대차 189%).
STABILITY_DISTORT_DEBT_RATIO = 150.0


def classify(induty_code, debt_ratio=None, corp_name: str = "") -> tuple[str, str]:
    """(트랙, 판정 사유). induty_code는 3자리·5자리가 섞여 온다."""
    code = str(induty_code or "").strip()
    name = str(corp_name or "")

    if code.startswith(KSIC_HOLDING):
        if debt_ratio is not None and debt_ratio > HOLDING_FINANCE_DEBT_RATIO:
            return config.TRACK_FINANCE, f"지주회사 + 부채비율 {debt_ratio:.0f}% → 금융지주"
        if debt_ratio is None and _FINANCE_NAME.search(name):
            return config.TRACK_FINANCE, "지주회사 + 상호에 금융 표기"
        return config.TRACK_HOLDING, "지주회사 (KSIC 64992)"

    if code[:2] in KSIC_FINANCE:
        return config.TRACK_FINANCE, f"KSIC {code} 금융·보험"
    if code[:2] in KSIC_PHARMA:
        return config.TRACK_PHARMA, f"KSIC {code} 의약품"

    # 업종코드가 제조인데 부채비율이 금융권 수준이면 코드를 의심한다.
    if debt_ratio is not None and debt_ratio > HOLDING_FINANCE_DEBT_RATIO * 2:
        return config.TRACK_FINANCE, f"부채비율 {debt_ratio:.0f}% — 업종코드 무시하고 금융"

    if not code:
        return config.TRACK_GENERAL, "업종코드 없음 — 기본 트랙"
    return config.TRACK_GENERAL, f"KSIC {code}"


def detect_finance_segment(raw_bs: pd.DataFrame, assets) -> tuple[bool, float]:
    """제조 + 금융 복합 여부.

    현대차·기아처럼 캐피탈을 연결하는 완성차는 **연결 부채비율이 금융 부문
    차입금 때문에 부풀려진다.** 실측: 현대차 연결 BS에 금융업채권 134조,
    운용리스자산 54조가 잡히고 부채비율이 189%인 반면, 캐피탈을 연결하지 않는
    기아는 62%다. 같은 산업인데 두 배 넘게 벌어진다.

    T1에 두되 이 플래그가 서면 안정성 지표를 세그먼트 기준으로 봐야 한다고
    표시한다. 세그먼트 데이터가 없으면 해당 지표는 NULL이지 추정값이 아니다.
    """
    if raw_bs is None or raw_bs.empty or not assets:
        return False, 0.0
    from .quarterly import parse_amount

    total = 0.0
    for r in raw_bs.itertuples():
        nm = str(getattr(r, "account_nm", "")).strip()
        if _FINANCE_ASSET.search(nm):
            v = parse_amount(getattr(r, "thstrm_amount", None))
            if v:
                total += abs(v)
    ratio = total / assets if assets else 0.0
    return ratio >= FINANCE_SEGMENT_MIN_RATIO, ratio


def assign(corp_code: str, ticker: str, corp_name: str, *,
           annual_row=None, raw_bs: pd.DataFrame | None = None) -> dict:
    """한 종목의 트랙 판정 결과."""
    from . import dart

    info = dart.company_info(corp_code)
    induty = info.get("induty_code")

    debt_ratio = None
    assets = None
    if annual_row is not None:
        liab, eq = annual_row.get("liabilities"), annual_row.get("equity")
        assets = annual_row.get("assets")
        if liab is not None and eq and pd.notna(liab) and pd.notna(eq) and eq > 0:
            debt_ratio = liab / eq * 100

    track, reason = classify(induty, debt_ratio, corp_name)
    has_fin, fin_ratio = detect_finance_segment(raw_bs, assets)
    has_fin = bool(has_fin and track == config.TRACK_GENERAL)

    # 금융성 자산이 있다 ≠ 안정성 지표가 망가졌다. 부채비율까지 부풀었을 때만
    # 세그먼트 기준으로 다시 봐야 하고, 세그먼트가 없으면 그 지표는 NULL이다.
    distorted = bool(has_fin and debt_ratio is not None
                     and debt_ratio > STABILITY_DISTORT_DEBT_RATIO)

    return {
        "ticker": ticker, "corp_code": corp_code, "corp_name": corp_name,
        "induty_code": str(induty or ""), "track": track, "track_reason": reason,
        "debt_ratio": debt_ratio,
        "has_finance_segment": has_fin,
        "stability_distorted": distorted,
        "finance_asset_ratio": fin_ratio,
    }


def null_metrics(track: str) -> set[str]:
    """이 트랙에서 계산하지 않는 지표. 값이 아니라 NULL이 정답이다."""
    return set(config.TRACK_NULL_METRICS.get(track, ()))
