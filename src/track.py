"""성과 추적 — 매달 후보 규칙들의 선택을 기록하고 다음 달에 정산한다.

왜 필요한가
    17·18번 백테스트는 전부 **인샘플**이다. 가중치를 그 기간 데이터로 맞췄고,
    12개 신호를 비교해 제일 좋은 것을 골랐다. 그래서 "성장성 부문이 좋다"는
    결론에 아직 손대지 않았다 — 근거로 쓰려면 오염되지 않은 표본이 필요하다.

    이 모듈은 오늘부터 **모든 후보 규칙의 선택을 동시에 기록**한다. 실제 매수는
    복합 점수로 하되, 나머지 규칙도 종이 위에서 같이 굴린다. 1년쯤 쌓이면
    "그때 성장성 부문으로 갔으면 어땠나"를 사후 해석이 아니라 기록으로 답할 수 있다.

기록 위치 (볼트)
    _성과추적/원장.csv       종목별 선택과 실현 수익률
    _성과추적/월별성과.csv    월별 규칙 성과 + 벤치마크 + 확산지수
    _성과추적/성과추적.md     사람이 읽는 누적 요약
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config

from . import store, strategies

FOLDER = "_성과추적"
LEDGER = "원장.csv"
MONTHLY = "월별성과.csv"
NOTE = "성과추적.md"

TOP_N = 5

# 규칙 이름 → (점수 계산, 게이트 적용 여부)
# 게이트를 쓰는 두 복합 규칙이 실제 볼트에 올라가는 것과 같다.
RULE_ORDER = [
    "복합:가치투자", "복합:성장주",
    "성장성 단독", "수익성 단독", "밸류에이션 단독", "PBR 단독",
    # 뉴스에 인용된 증권사 목표가 (21_news_consensus.py). 과거 데이터가 없어
    # 백테스트가 불가능하므로, 점수에 넣지 않고 여기서 종이로만 굴린다.
    "컨센 상승여력", "컨센 상향비율", "복합성장∩상향",
]


def _weighted(df: pd.DataFrame, w: dict[str, float]) -> pd.Series:
    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    for cat, weight in w.items():
        if not weight:
            continue
        s = pd.to_numeric(df[f"{cat}_score"], errors="coerce")
        ok = s.notna()
        num = num + s.fillna(0) * weight * ok
        den = den + weight * ok
    return pd.Series(np.where(den > 0, num / den, np.nan), index=df.index)


MIN_BROKERS = 3      # 증권사가 이보다 적으면 컨센서스 규칙에서 제외


def rule_scores(df: pd.DataFrame) -> dict[str, pd.Series]:
    """규칙별 점수. 높을수록 상위. 게이트는 selections()에서 따로 건다.

    컨센서스 규칙은 df에 upside·revision·n_brokers 컬럼이 있을 때만 값이 찬다
    (없으면 전부 결측 → 그 규칙은 그 달에 선택 없음).
    """
    pbr = pd.to_numeric(df["pbr"], errors="coerce").where(lambda x: x > 0)
    nb = pd.to_numeric(df.get("n_brokers"), errors="coerce") if "n_brokers" in df else None
    enough = nb >= MIN_BROKERS if nb is not None else pd.Series(False, index=df.index)
    up = pd.to_numeric(df.get("upside"), errors="coerce").where(enough) \
        if "upside" in df else pd.Series(np.nan, index=df.index)
    rev = pd.to_numeric(df.get("revision"), errors="coerce").where(enough) \
        if "revision" in df else pd.Series(np.nan, index=df.index)
    growth = _weighted(df, strategies.WEIGHTS[strategies.GROWTH])
    return {
        "복합:가치투자": _weighted(df, strategies.WEIGHTS[strategies.VALUE]),
        "복합:성장주": _weighted(df, strategies.WEIGHTS[strategies.GROWTH]),
        "성장성 단독": pd.to_numeric(df["growth_score"], errors="coerce"),
        "수익성 단독": pd.to_numeric(df["profitability_score"], errors="coerce"),
        "밸류에이션 단독": pd.to_numeric(df["valuation_score"], errors="coerce"),
        # 낮을수록 좋은 지표는 100에서 빼서 뒤집는다.
        # rank(ascending=False)로 쓰면 방향을 헷갈리기 쉽다.
        "PBR 단독": 100 - pbr.rank(pct=True) * 100,
        "컨센 상승여력": up,
        # 같은 상향비율이면 증권사가 많은 쪽을 위로 (1,000으로 나눠 순위만 가른다)
        "컨센 상향비율": rev + (nb.fillna(0) / 1000 if nb is not None else 0),
        # 우리 점수와 증권사 시각이 같은 방향인 종목 — 상향 우위인 것만 남긴다
        "복합성장∩상향": growth.where(rev > 0),
    }


GATED = {"복합:가치투자": strategies.VALUE, "복합:성장주": strategies.GROWTH,
         "복합성장∩상향": strategies.GROWTH}


def selections(df: pd.DataFrame, gate_pass: dict[str, pd.Series],
               top: int = TOP_N) -> pd.DataFrame:
    """규칙별 상위 top종목. 반환 컬럼: rule, rank, ticker, name, score."""
    rows = []
    for rule, s in rule_scores(df).items():
        d = df.assign(sc=s)
        if rule in GATED:
            d = d[gate_pass[GATED[rule]].reindex(d.index).fillna(False)]
        # itertuples는 밑줄로 시작하는 컬럼명을 위치 이름으로 바꿔버린다
        d = d[d["sc"].notna()].nlargest(top, "sc")
        for i, (_, r) in enumerate(d.iterrows(), 1):
            rows.append({"rule": rule, "rank": i, "ticker": str(r["ticker"]),
                         "name": str(r["name"]), "score": round(float(r["sc"]), 2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 원장 입출력
# ---------------------------------------------------------------------------
def _dir(root: Path) -> Path:
    p = Path(root) / FOLDER
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_csv(root: Path, name: str, cols: list[str]) -> pd.DataFrame:
    p = _dir(root) / name
    if not p.exists():
        return pd.DataFrame(columns=cols)
    d = pd.read_csv(p, dtype={"ticker": str})
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    return d[cols]


def write_csv(root: Path, name: str, df: pd.DataFrame) -> Path:
    p = _dir(root) / name
    # 엑셀에서 바로 열리도록 BOM을 붙인다
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return p


LEDGER_COLS = ["asof", "rule", "rank", "ticker", "name", "score",
               "entry_price", "exit_asof", "exit_price", "ret_pct"]
MONTHLY_COLS = ["asof", "exit_asof", "days", "bench_eq", "bench_cap",
                "diffusion", "n_universe"] + RULE_ORDER


def close_at(px: pd.DataFrame, asof: str) -> pd.Series:
    """기준일 이하 마지막 수정종가."""
    v = px[pd.to_datetime(px["date"]) <= pd.Timestamp(asof)]
    if v.empty:
        return pd.Series(dtype=float)
    return v.sort_values("date").groupby("ticker")["close_adj"].last()


MIN_HOLD_DAYS = 15   # 이보다 짧으면 "같은 달 재실행"으로 본다


def _held_days(entry: pd.Series, asof: str) -> pd.Series:
    return (pd.Timestamp(asof) - pd.to_datetime(entry)).dt.days


def settle(root: Path, px: pd.DataFrame, asof: str) -> tuple[int, list[str]]:
    """미정산 건을 현재 기준일 종가로 정산한다.

    보유일수가 MIN_HOLD_DAYS 미만이면 정산하지 않는다. 같은 달에 두 번 돌렸을 때
    며칠짜리 수익률이 한 달치로 기록되면 표본이 오염된다 — 그런 건은
    record()가 덮어쓰도록 남겨둔다.
    """
    led = load_csv(root, LEDGER, LEDGER_COLS)
    if led.empty:
        return 0, []
    open_rows = (led["ret_pct"].isna()
                 & (led["asof"] < asof)
                 & (_held_days(led["asof"], asof) >= MIN_HOLD_DAYS))
    if not open_rows.any():
        return 0, []
    now = close_at(px, asof)
    led.loc[open_rows, "exit_asof"] = asof
    led.loc[open_rows, "exit_price"] = led.loc[open_rows, "ticker"].map(now)
    ok = open_rows & led["exit_price"].notna() & led["entry_price"].notna()
    led.loc[ok, "ret_pct"] = (
        led.loc[ok, "exit_price"] / led.loc[ok, "entry_price"] - 1) * 100
    led["ret_pct"] = led["ret_pct"].round(2)
    write_csv(root, LEDGER, led)
    months = sorted(led.loc[ok, "asof"].unique())
    return int(ok.sum()), list(months)


def record(root: Path, asof: str, sel: pd.DataFrame, px: pd.DataFrame) -> int:
    """이번 달 선택을 원장에 추가한다.

    같은 기준일, 또는 MIN_HOLD_DAYS 안에 있는 **미정산** 기록은 덮어쓴다.
    같은 달을 두 번 돌려도 한 달에 한 줄만 남게 하기 위해서다.
    이미 정산된 기록은 건드리지 않는다.
    """
    led = load_csv(root, LEDGER, LEDGER_COLS)
    if not led.empty:
        stale = (led["ret_pct"].isna()
                 & (_held_days(led["asof"], asof).abs() < MIN_HOLD_DAYS))
        led = led[~stale]
    now = close_at(px, asof)
    add = sel.copy()
    add.insert(0, "asof", asof)
    add["entry_price"] = add["ticker"].map(now)
    for c in ("exit_asof", "exit_price", "ret_pct"):
        add[c] = np.nan
    add = add[LEDGER_COLS]
    out = add if led.empty else pd.concat([led, add], ignore_index=True)
    out = out.sort_values(["asof", "rule", "rank"])
    write_csv(root, LEDGER, out)
    return len(add)


def close_month(root: Path, entry_asof: str, exit_asof: str,
                px: pd.DataFrame, snapshot: pd.DataFrame,
                diffusion: float | None) -> dict | None:
    """정산된 달의 월별 성과 한 줄을 만든다."""
    led = load_csv(root, LEDGER, LEDGER_COLS)
    g = led[(led["asof"] == entry_asof) & led["ret_pct"].notna()]
    if g.empty:
        return None
    p0, p1 = close_at(px, entry_asof), close_at(px, exit_asof)
    uni = snapshot["ticker"].astype(str)
    r = (p1.reindex(uni) / p0.reindex(uni) - 1) * 100
    cap = pd.to_numeric(snapshot.set_index("ticker")["market_cap"],
                        errors="coerce").reindex(uni)
    m = r.notna() & cap.notna()
    row = {"asof": entry_asof, "exit_asof": exit_asof,
           "days": (pd.Timestamp(exit_asof) - pd.Timestamp(entry_asof)).days,
           "bench_eq": round(float(r[m].mean()), 2),
           "bench_cap": round(float(np.average(r[m], weights=cap[m])), 2),
           "diffusion": round(diffusion * 100, 1) if diffusion is not None else np.nan,
           "n_universe": int(m.sum())}
    for rule in RULE_ORDER:
        v = g[g["rule"] == rule]["ret_pct"]
        row[rule] = round(float(v.mean()), 2) if len(v) else np.nan

    mon = load_csv(root, MONTHLY, MONTHLY_COLS)
    mon = mon[mon["asof"] != entry_asof]
    new = pd.DataFrame([row])
    mon = new if mon.empty else pd.concat([mon, new], ignore_index=True)
    write_csv(root, MONTHLY, mon.sort_values("asof"))
    return row


def diffusion_index(ttm: pd.DataFrame, asof: str,
                    tickers: list[str]) -> float | None:
    """매출성장 확산지수 — 최근 4분기 중 3회 이상 매출이 늘어난 기업의 비율.

    97개월 검증에서 이후 3개월 시장수익률과 순위상관 +0.48로 나온 지표.
    직전 시장수익률을 통제해도 남았지만 표본이 작아 아직 매매 근거는 아니다.
    지금부터 기록해 아웃오브샘플을 쌓는다.
    """
    gf = strategies.gate_features(ttm, asof)
    if gf.empty:
        return None
    gf = gf[gf["ticker"].isin(tickers)]
    if gf.empty:
        return None
    return float((gf["rev_growth_q_of_4"] >= 3).mean())
