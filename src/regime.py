"""시장 국면 판별 — 점수 하락이 그 회사 탓인가, 시장 탓인가.

전쟁·유가 급등·금리 발작 같은 사건을 재무 데이터로 직접 알아낼 수는 없다.
대신 **그 사건이 남기는 흔적**은 잡을 수 있다 — 시장 요인이면 여러 종목이
동시에 같은 방향으로 나빠지고, 개별 요인이면 그 종목만 나빠진다.

그래서 원인을 이름 붙이려 하지 않고 **폭(breadth)** 을 본다.
같이 무너졌으면 매도를 보류하고, 혼자 무너졌으면 매도한다.

보류는 무한정이 아니다. 시장 요인이라며 계속 들고 있으면 진짜 부실을 놓친다.
"""
from __future__ import annotations

import pandas as pd

import config

# 판별 임계값
DROP_PT = 5.0            # 이만큼 떨어지면 "악화"로 센다
BREADTH_MARKET = 0.40    # 트랙의 이 비율 이상이 동반 악화하면 시장 요인
BREADTH_SECTOR = 0.50    # 업종의 이 비율 이상이면 업종 요인
FUNDAMENTAL_TOL = 0.25   # 절대 지표가 이 비율 넘게 나빠졌으면 회사 문제로 본다
MAX_DEFER_MONTHS = 6     # 보류 상한 — 넘으면 시장 요인이어도 매도

# 절대 지표로 회사의 실질 훼손을 판별할 때 쓰는 항목
CORE = [("roe", +1), ("operating_margin", +1), ("ocf_to_ni", +1), ("debt_ratio", -1)]


def breadth(prev: pd.DataFrame, now: pd.DataFrame,
            score_col: str = "quality_score") -> dict:
    """전월 대비 점수가 떨어진 종목의 비율을 트랙·업종별로.

    prev/now: grades 스냅샷 (ticker, track, sector, {score_col})
    """
    if prev is None or prev.empty or now is None or now.empty:
        return {"track": {}, "sector": {}, "overall": 0.0, "n": 0}
    keys = ["ticker", "track", score_col] + (["sector"] if "sector" in now.columns else [])
    a = prev[[c for c in keys if c in prev.columns]].rename(columns={score_col: "prev"})
    b = now[[c for c in keys if c in now.columns]].rename(columns={score_col: "now"})
    m = a[["ticker", "prev"]].merge(b, on="ticker", how="inner").dropna(
        subset=["prev", "now"])
    if m.empty:
        return {"track": {}, "sector": {}, "overall": 0.0, "n": 0}
    m["worse"] = (m["now"] - m["prev"]) <= -DROP_PT
    out = {
        "overall": float(m["worse"].mean()),
        "n": int(len(m)),
        "track": m.groupby("track")["worse"].mean().to_dict(),
        "sector": (m.groupby("sector")["worse"].mean().to_dict()
                   if "sector" in m.columns else {}),
    }
    return out


def fundamentals_held_up(row: dict, prev_row: dict | None) -> tuple[bool, list[str]]:
    """절대 지표가 버티고 있는가. 점수(상대 순위)와 별개로 본다.

    시장이 통째로 빠지면 순위는 유지되는데도 게이트에 걸릴 수 있고,
    반대로 동종업체가 좋아져서 순위만 밀릴 수도 있다. 그래서 회사 자체의
    절대 수치가 실제로 나빠졌는지 따로 확인한다.
    """
    if not prev_row:
        return True, []
    hurt = []
    for col, direction in CORE:
        a, b = prev_row.get(col), row.get(col)
        if a is None or b is None or pd.isna(a) or pd.isna(b) or a == 0:
            continue
        change = (b - a) / abs(a) * direction
        if change < -FUNDAMENTAL_TOL:
            hurt.append(f"{col} {a:.1f} → {b:.1f}")
    return (not hurt), hurt


def defer_sell(row: dict, prev_row: dict | None, bd: dict,
               defer_months: int = 0) -> dict:
    """매도를 보류할 것인가.

    보류 조건 (하나라도 해당하면 보류):
      ① 같은 트랙의 40% 이상이 동반 악화 — 시장 요인
      ② 같은 업종의 50% 이상이 동반 악화 — 업종 요인
      ③ 점수는 떨어졌지만 절대 지표(ROE·영업이익률·영업CF/순이익·부채비율)는 유지

    단, 누적 보류가 6개월을 넘으면 보류하지 않는다.
    """
    if defer_months >= MAX_DEFER_MONTHS:
        return {"defer": False, "reason": "",
                "note": f"보류 {defer_months}개월 누적 — 상한({MAX_DEFER_MONTHS}개월) 초과로 매도 집행"}

    track = str(row.get("track", ""))
    sector = str(row.get("sector", ""))
    t_share = bd.get("track", {}).get(track)
    s_share = bd.get("sector", {}).get(sector)

    if t_share is not None and t_share >= BREADTH_MARKET:
        return {"defer": True,
                "reason": f"시장 요인 — 같은 트랙({track})의 {t_share*100:.0f}%가 동반 악화",
                "note": "개별 기업 문제로 보기 어려움. 다음 달 재평가"}
    if s_share is not None and s_share >= BREADTH_SECTOR:
        return {"defer": True,
                "reason": f"업종 요인 — 같은 업종의 {s_share*100:.0f}%가 동반 악화",
                "note": "업종 전반의 문제. 다음 달 재평가"}

    ok, hurt = fundamentals_held_up(row, prev_row)
    if ok:
        return {"defer": True,
                "reason": "절대 지표 유지 — 순위만 밀렸고 회사 수치는 버티는 중",
                "note": "동종업체가 좋아져 상대 순위가 내려간 경우"}

    return {"defer": False,
            "reason": "",
            "note": "개별 악화 — " + " · ".join(hurt) if hurt else "개별 악화"}


def applies_to(strategy: str) -> bool:
    """보류 규칙은 장기 전략에만 적용한다.

    급등주는 애초에 2~8주 보유를 전제한 전략이라, 시장 요인이든 개별 요인이든
    모멘텀이 꺾이면 나가는 게 맞다. 여기에 보류를 넣으면 전략의 전제가 깨진다.
    """
    return strategy in ("value", "growth")
