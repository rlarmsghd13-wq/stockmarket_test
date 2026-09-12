"""계정 표준화 — DART 원문 계정을 우리 필드명으로 매핑.

회사마다 계정과목명이 다르다. IFRS 표준 태그(account_id)를 1순위로 쓰고,
태그가 비어 있을 때만(`-` 로 오는 경우가 많다) 계정명 정규식으로 폴백한다.

**매칭 실패는 NULL로 두고 로그에 남긴다. 비슷한 이름으로 추측해 채우지 않는다.**
잘못 채운 값 하나가 그 종목의 지표 전체를 조용히 망친다.
"""
from __future__ import annotations

import re

# sj_div: BS(재무상태표) IS(손익) CIS(포괄손익) CF(현금흐름표)
# (필드명, sj_div들, account_id 후보, 계정명 정규식 후보)
SPEC: list[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = [
    # ---- 손익계산서 (기간값) ----
    ("revenue", ("IS", "CIS"),
     ("ifrs-full_Revenue", "ifrs-full_RevenueFromContractsWithCustomers",
      "ifrs_Revenue"),
     (r"^매출액$", r"^수익\(매출액\)$", r"^영업수익$", r"^매출$")),

    ("gross_profit", ("IS", "CIS"),
     ("ifrs-full_GrossProfit", "ifrs_GrossProfit"),
     (r"^매출총이익",)),

    ("operating_income", ("IS", "CIS"),
     ("dart_OperatingIncomeLoss", "ifrs-full_ProfitLossFromOperatingActivities",
      "ifrs_ProfitLossFromOperatingActivities"),
     (r"^영업이익", r"^영업손실")),

    ("pretax_income", ("IS", "CIS"),
     ("ifrs-full_ProfitLossBeforeTax", "ifrs_ProfitLossBeforeTax"),
     (r"법인세.*차감전.*(순)?이익", r"^세전.*이익")),

    ("tax_expense", ("IS", "CIS"),
     ("ifrs-full_IncomeTaxExpenseContinuingOperations",
      "ifrs_IncomeTaxExpenseContinuingOperations"),
     (r"^법인세비용",)),

    ("net_income", ("IS", "CIS"),
     ("ifrs-full_ProfitLoss", "ifrs_ProfitLoss"),
     (r"^당기순이익", r"^당기순손실", r"^분기순이익", r"^반기순이익")),

    ("net_income_parent", ("IS", "CIS"),
     ("ifrs-full_ProfitLossAttributableToOwnersOfParent",
      "ifrs_ProfitLossAttributableToOwnersOfParent"),
     (r"지배기업.*소유주.*지분", r"^지배주주")),

    # ---- 재무상태표 (시점값) ----
    ("assets", ("BS",), ("ifrs-full_Assets", "ifrs_Assets"), (r"^자산총계$",)),
    ("current_assets", ("BS",), ("ifrs-full_CurrentAssets", "ifrs_CurrentAssets"),
     (r"^유동자산$",)),
    ("liabilities", ("BS",), ("ifrs-full_Liabilities", "ifrs_Liabilities"),
     (r"^부채총계$",)),
    ("current_liabilities", ("BS",),
     ("ifrs-full_CurrentLiabilities", "ifrs_CurrentLiabilities"),
     (r"^유동부채$",)),
    ("equity", ("BS",), ("ifrs-full_Equity", "ifrs_Equity"), (r"^자본총계$",)),
    ("equity_parent", ("BS",),
     ("ifrs-full_EquityAttributableToOwnersOfParent",
      "ifrs_EquityAttributableToOwnersOfParent"),
     (r"지배기업.*소유주.*지분", r"^지배주주지분")),
    ("cash", ("BS",),
     ("ifrs-full_CashAndCashEquivalents", "ifrs_CashAndCashEquivalents"),
     (r"^현금및현금성자산",)),

    # 신종자본증권(하이브리드) — 회계상 자본이지만 보통주 몫이 아니다.
    # 금융지주에 특히 크다. KB금융 2025년 지배주주지분 59.05조 중 4.36조가 이것이고,
    # 빼지 않으면 BPS가 8% 과대평가된다 (빼면 KRX 공표값과 0.1% 이내).
    ("hybrid_capital", ("BS",), (),
     (r"^신종자본증권", r"하이브리드채권", r"^기타자본증권$")),

    # ---- 현금흐름표 (기간값) ----
    ("ocf", ("CF",),
     ("ifrs-full_CashFlowsFromUsedInOperatingActivities",
      "ifrs_CashFlowsFromUsedInOperatingActivities"),
     (r"영업활동.*현금흐름",)),

    # CAPEX·감가상각은 표준 태그가 거의 없어 계정명 매칭에 의존한다.
    ("capex_tangible", ("CF",), (),
     (r"유형자산.*취득", r"유형자산의\s*증가")),
    ("capex_intangible", ("CF",), (),
     (r"무형자산.*취득", r"무형자산의\s*증가")),
    # 감가상각은 현금흐름표의 '조정' 항목으로 들어가고 표준 태그가 거의 없다.
    # 회사마다 표기가 제각각이라 앞머리 고정(^)으로 잡으면 대부분 놓친다.
    ("depreciation", ("CF", "IS", "CIS"), (),
     (r"감가상각", r"상각비용")),
    ("amortization", ("CF", "IS", "CIS"), (),
     (r"무형자산상각", r"무형자산의상각")),
]

# CAPEX는 현금흐름표에서 음수(유출)로 오는 경우와 양수로 오는 경우가 섞여 있다.
# 계산 단계에서 abs()를 취한다.
CAPEX_FIELDS = ("capex_tangible", "capex_intangible")

_COMPILED = [
    (field, sj, ids, tuple(re.compile(p) for p in pats))
    for field, sj, ids, pats in SPEC
]

ALL_FIELDS = [s[0] for s in SPEC]
# 기간값 — TTM 합산 대상. 나머지는 시점값이라 최근 분기말을 그대로 쓴다.
FLOW_FIELDS = {
    "revenue", "gross_profit", "operating_income", "pretax_income",
    "tax_expense", "net_income", "net_income_parent", "ocf",
    "capex_tangible", "capex_intangible", "depreciation", "amortization",
}
STOCK_FIELDS = set(ALL_FIELDS) - FLOW_FIELDS


def _clean_name(s: str) -> str:
    return re.sub(r"[\s()]", "", str(s or ""))


# 표준 태그 → 필드 (전 필드 통합). 이름 폴백보다 항상 우선한다.
KNOWN_IDS: dict[str, str] = {aid: field for field, _, ids, _ in SPEC for aid in ids}
FIELD_SJ: dict[str, tuple[str, ...]] = {field: sjs for field, sjs, _, _ in SPEC}

# 절대 매칭하면 안 되는 계정.
#
# 포괄손익계산서에는 당기순이익과 **이름이 완전히 같은** 총포괄이익 행이 있다.
# 예: 삼성물산 2025년 사업보고서에 "지배기업 소유주지분"이 두 번 나오는데
#     하나는 ComprehensiveIncomeAttributableToOwnersOfParent (19.29조),
#     하나는 ProfitLossAttributableToOwnersOfParent (2.44조)다.
# 이름만 보면 구분이 불가능하고, 앞줄을 집으면 순이익이 8배로 부풀어
# EPS·PER·ROE가 통째로 틀린다. 태그로만 걸러낼 수 있다.
EXCLUDE_ID_TOKENS = ("ComprehensiveIncome", "OtherComprehensive")
EXCLUDE_NAME = re.compile(r"포괄|주당")


def match_field(account_id: str, account_nm: str, sj_div: str) -> str | None:
    """한 계정 행이 어느 표준 필드인지. 못 찾으면 None.

    우선순위: ① IFRS 표준 태그 정확 일치 → ② 배제 규칙 → ③ 계정명 정규식.
    태그가 있는데 우리 목록에 없다면, 그건 '다른 개념'일 가능성이 높으므로
    이름이 비슷하다는 이유로 끌어오지 않는다.
    """
    aid = (account_id or "").strip()
    nm = _clean_name(account_nm)

    if aid in KNOWN_IDS:
        field = KNOWN_IDS[aid]
        return field if sj_div in FIELD_SJ[field] else None

    if aid and any(tok in aid for tok in EXCLUDE_ID_TOKENS):
        return None
    if EXCLUDE_NAME.search(nm):
        return None

    for field, sjs, ids, pats in _COMPILED:
        if sj_div not in sjs:
            continue
        for p in pats:
            if p.search(nm):
                return field
    return None
