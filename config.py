"""전역 설정 — 경로, 임계값, 상수.

임계값은 여기 한 곳에만 둔다. 백테스트에서 트랙별로 교정할 때
코드 여기저기를 뒤지지 않기 위해서다.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# 경로
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"          # API 원본 응답 (파싱 로직 고칠 때 재수집 안 하려고)
CURATED = DATA / "curated"  # 정제 결과
LOGS = ROOT / "logs"

for _p in (DATA, RAW, RAW / "dart", CURATED, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# 수집 범위
# --------------------------------------------------------------------------
# 백테스트를 포함하므로 과거 시점 유니버스를 복원해야 한다.
# 5년이면 코로나 급락(2020)은 빠지지만 2022 하락장과 2023~ 회복장은 들어온다.
# 더 길게 잡으면 수집 시간이 비례해 늘고, 그 시절 사업구조와 지금이 달라
# 지표의 의미도 흐려진다.
BACKTEST_START = "2021-01-01"

# 재무 수집 연도 범위. TTM은 전년 연간이 있어야 계산되므로 1년 더 앞서 받는다.
FIN_YEARS_BACK = 6
UNIVERSE_SIZE = 200
MIN_MARKET_CAP = 5_000e8          # 5,000억
MIN_AVG_TRADING_VALUE = 50e8      # 60일 일평균 거래대금 50억
MIN_LISTED_YEARS = 3

MARKETS = ("KOSPI", "KOSDAQ")

# --------------------------------------------------------------------------
# 트랙
# --------------------------------------------------------------------------
TRACK_GENERAL = "T1"
TRACK_FINANCE = "T2"
TRACK_PHARMA = "T3"
TRACK_HOLDING = "T4"

# 재무 특성 기반 오버라이드 임계값 (업종코드보다 우선)
FINANCE_REVENUE_RATIO = 0.50   # 금융수익 / 총수익
PHARMA_RND_RATIO = 0.20        # R&D / 매출
HOLDING_EQUITY_INCOME_RATIO = 0.70  # 별도 배당·지분법이익 / 별도 수익

# 트랙별로 계산하지 않는 지표 — 계산 시 NULL로 둔다.
TRACK_NULL_METRICS = {
    TRACK_FINANCE: {
        "debt_ratio", "current_ratio", "fcf_ttm", "fcf_to_ni",
        "operating_margin", "ev_ebitda", "net_debt", "net_debt_ebitda",
    },
    TRACK_PHARMA: {
        "per", "peg", "roe", "operating_margin", "normalized_per",
    },
    TRACK_HOLDING: {
        "per", "revenue_growth_yoy",
    },
}

# --------------------------------------------------------------------------
# 적자 처리 (영업이익 기준, T3 제외)
# --------------------------------------------------------------------------
LOSS_TEMP = 1        # 1년 적자 → 정상화 이익으로 대체 평가
LOSS_DOUBT = 2       # 2년 연속 → 감점 + 가치 게이트 탈락
LOSS_FAIL = 3        # 3년 연속 → 유니버스 제외
CRITICAL_DEBT_RATIO = 200.0   # 적자 + 부채비율 초과 + 영업CF 음수 → 즉시 제외

# T3 캐시 런웨이 (분기)
RUNWAY_DANGER = 4
RUNWAY_WARN = 8

# --------------------------------------------------------------------------
# 옵시디언 볼트
# --------------------------------------------------------------------------
# 사용자의 기존 주식 볼트(구글 드라이브 동기화) 안에 별도 최상위 폴더로 둔다.
# 손으로 쓴 차트 분석(`1. 종목 분석`)과 성격이 다르므로 섞지 않는다.
# 자동 생성은 이 폴더 안에서만 일어난다.
# 슬래시로 적는다. 윈도우 경로를 백슬래시로 쓰면 "" 같은 조각이
# 이스케이프로 먹혀 조용히 다른 경로가 된다. 파이썬은 슬래시를 그대로 받는다.
# 환경마다 다르므로 .env 의 VAULT_ROOT 로 덮어쓸 수 있게 한다.
# 슬래시로 적는다. 윈도우 경로를 백슬래시로 쓰면 "\4" 같은 조각이 이스케이프로
# 먹혀 조용히 다른 경로가 된다. 파이썬은 슬래시를 그대로 받는다.
VAULT_ROOT = os.environ.get("VAULT_ROOT") or str(ROOT / "vault")

# --------------------------------------------------------------------------
# 판정 임계값 — 매수는 점수만으로 결정한다
# --------------------------------------------------------------------------
# 신규 매수 기준과 보유 유지 기준을 다르게 둔다(히스테리시스).
# 같은 값을 쓰면 79↔77을 오가며 매달 매수·매도 신호가 뒤집히고
# 거래비용만 쌓인다. 한 번 산 종목은 62점까지는 들고 간다.
SCORE_BUY = 78.0        # 신규 매수
SCORE_HOLD = 62.0       # 보유 유지 (미보유면 관망)
SCORE_TRIM = 52.0       # 비중 축소
# 그 미만은 매도

# --------------------------------------------------------------------------
# 계산 상수
# --------------------------------------------------------------------------
DEFAULT_TAX_RATE = 0.22       # 실효세율 이상치일 때 대체값
TAX_RATE_BOUNDS = (0.0, 0.50)  # 이 범위 밖이면 DEFAULT 사용
WINSOR_PCT = 0.01             # 상하위 1% 클리핑
NORMALIZE_QUARTERS = 20       # 정상화 이익 = 최근 20분기 평균 영업이익률
GROWTH_VOL_QUARTERS = 8       # 성장 변동성 계산 구간

# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
DART_KEY = os.environ.get("DART_API_KEY", "")
DART_BASE = "https://opendart.fss.or.kr/api"
DART_RATE_LIMIT_PER_MIN = 900   # 공식 한도 20,000/일. 분당은 보수적으로.
DART_RETRY = 3
DART_TIMEOUT = 20

# 분기보고서 코드
REPRT_Q1 = "11013"
REPRT_H1 = "11012"
REPRT_Q3 = "11014"
REPRT_ANNUAL = "11011"
REPRT_ORDER = [REPRT_Q1, REPRT_H1, REPRT_Q3, REPRT_ANNUAL]
REPRT_QUARTER_NO = {REPRT_Q1: 1, REPRT_H1: 2, REPRT_Q3: 3, REPRT_ANNUAL: 4}


def require_dart_key() -> str:
    if not DART_KEY:
        raise RuntimeError(
            "DART_API_KEY가 없습니다.\n"
            "  1) https://opendart.fss.or.kr 에서 인증키를 발급하세요 (무료).\n"
            "  2) screener/.env 에 DART_API_KEY=발급받은키 한 줄을 추가하세요.\n"
            "  3) 다시 실행하세요."
        )
    return DART_KEY
