"""DART OpenAPI 클라이언트.

원칙
  - 응답 원본 JSON을 그대로 디스크에 남긴다. 파싱 로직을 고칠 때 재수집하지 않기 위해서다.
  - 모든 재무 레코드에 접수일(rcept_dt)을 붙인다. 시점 잠금의 근거가 된다.
"""
from __future__ import annotations

import io
import time
import zipfile
from xml.etree import ElementTree

import pandas as pd
import requests

import config
from . import env, store

env.load()

# 보고서명 안의 결산기 표기 — 사업연도를 확정하는 데 쓴다.
PERIOD_END = {
    config.REPRT_Q1: "03",
    config.REPRT_H1: "06",
    config.REPRT_Q3: "09",
    config.REPRT_ANNUAL: "12",
}

_SESSION = requests.Session()
_last_call = 0.0
_MIN_INTERVAL = 60.0 / config.DART_RATE_LIMIT_PER_MIN


def _throttle() -> None:
    global _last_call
    wait = _MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def _get(endpoint: str, **params) -> dict:
    params["crtfc_key"] = config.require_dart_key()
    url = f"{config.DART_BASE}/{endpoint}"
    last = None
    for i in range(config.DART_RETRY):
        _throttle()
        try:
            r = _SESSION.get(url, params=params, timeout=config.DART_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"DART 호출 실패: {endpoint} {params.get('corp_code')}") from last


# ---------------------------------------------------------------------------
# 기업 코드 매핑 (corp_code ↔ ticker)
# ---------------------------------------------------------------------------
def corp_codes(refresh: bool = False) -> pd.DataFrame:
    """corpCode.xml — DART corp_code와 종목코드(ticker) 매핑.

    상장폐지 종목도 여기 남아 있어 과거 재무 조회가 가능하다. 생존편향 제거의 전제.
    """
    cached = config.RAW / "dart" / "corp_codes.parquet"
    if cached.exists() and not refresh:
        return pd.read_parquet(cached)

    _throttle()
    r = _SESSION.get(
        f"{config.DART_BASE}/corpCode.xml",
        params={"crtfc_key": config.require_dart_key()},
        timeout=60,
    )
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        xml = zf.read(zf.namelist()[0])

    rows = []
    for el in ElementTree.fromstring(xml).iter("list"):
        stock_code = (el.findtext("stock_code") or "").strip()
        rows.append({
            "corp_code": (el.findtext("corp_code") or "").strip(),
            "corp_name": (el.findtext("corp_name") or "").strip(),
            "ticker": stock_code,
            "modify_date": (el.findtext("modify_date") or "").strip(),
        })
    df = pd.DataFrame(rows)
    df = df[df["ticker"].str.len() == 6].reset_index(drop=True)  # 상장사만
    df.to_parquet(cached, index=False)
    return df


# ---------------------------------------------------------------------------
# 재무제표 원문
# ---------------------------------------------------------------------------
def financials(corp_code: str, year: int, reprt_code: str,
               fs_div: str = "CFS") -> pd.DataFrame:
    """단일회사 전체 재무제표 (fnlttSinglAcntAll).

    fs_div: CFS(연결) 우선. 연결이 없는 회사는 호출부에서 OFS로 재시도한다.
    반환 컬럼에는 rcept_no가 포함되며, 접수일은 `filing_date()`로 따로 조회한다.
    """
    key = ("dart", "fin", f"{corp_code}_{year}_{reprt_code}_{fs_div}")
    cached = store.load_raw_json(*key)
    if cached is None:
        cached = _get("fnlttSinglAcntAll.json", corp_code=corp_code,
                      bsns_year=str(year), reprt_code=reprt_code, fs_div=fs_div)
        store.save_raw_json(cached, *key)

    status = cached.get("status")
    if status != "000":
        # 013 = 조회 데이터 없음. 정상적인 결과이므로 예외로 만들지 않는다.
        return pd.DataFrame()

    df = pd.DataFrame(cached.get("list", []))
    if df.empty:
        return df
    df["corp_code"] = corp_code
    df["bsns_year"] = year
    df["reprt_code"] = reprt_code
    df["fs_div"] = fs_div
    return df


def company_info(corp_code: str) -> dict:
    """기업개황 (company.json) — 표준산업분류코드(induty_code)가 트랙 판정의 1차 근거."""
    key = ("dart", "company", corp_code)
    cached = store.load_raw_json(*key)
    if cached is None:
        cached = _get("company.json", corp_code=corp_code)
        store.save_raw_json(cached, *key)
    return cached if cached.get("status") == "000" else {}


def treasury_shares(corp_code: str, year: int,
                    reprt_code: str = config.REPRT_ANNUAL) -> int | None:
    """기말 자기주식 보유 수량 (tesstkAcqsDspsSttus).

    EPS·BPS의 분모는 상장주식수가 아니라 **유통주식수(상장 − 자기주식)**다.
    자기주식은 의결권도 배당권도 없다. KRX 공표값도 이 기준이며,
    자사주 비중이 큰 기업에서 차이가 크게 벌어진다 —
    SK는 자사주 24.6%라 분모를 안 빼면 BPS가 32.6% 낮게 나온다.
    """
    key = ("dart", "treasury", f"{corp_code}_{year}_{reprt_code}")
    cached = store.load_raw_json(*key)
    if cached is None:
        cached = _get("tesstkAcqsDspsSttus.json", corp_code=corp_code,
                      bsns_year=str(year), reprt_code=reprt_code)
        store.save_raw_json(cached, *key)
    if cached.get("status") != "000":
        return None

    total = 0
    found = False
    for r in cached.get("list", []):
        method = str(r.get("acqs_mth1", "")).strip()
        # 표에 '총계' 행이 있고 그 아래 세부 행이 반복된다. 총계만 더한다.
        if "총" not in method or "계" not in method:
            continue
        v = str(r.get("trmend_qy", "")).replace(",", "").strip()
        if v.lstrip("-").isdigit():
            total += int(v)
            found = True
    return total if found else None


def filing_date(corp_code: str, year: int, reprt_code: str) -> str | None:
    """보고서 접수일(rcept_dt) — 시점 잠금의 기준.

    이 값이 없으면 그 레코드는 백테스트에서 쓸 수 없다.
    """
    key = ("dart", "filing", f"{corp_code}_{year}_{reprt_code}_v2")
    cached = store.load_raw_json(*key)
    if cached is None:
        pblntf_detail = {
            config.REPRT_ANNUAL: "A001",   # 사업보고서
            config.REPRT_H1: "A002",       # 반기보고서
            config.REPRT_Q1: "A003",       # 분기보고서
            config.REPRT_Q3: "A003",
        }[reprt_code]
        # 사업연도 Y의 보고서는 Y 안에 제출되기도 하고(1Q/반기/3Q),
        # 이듬해 3월에 제출되기도 한다(사업보고서). 넉넉히 훑고 아래에서 걸러낸다.
        cached = _get("list.json", corp_code=corp_code,
                      bgn_de=f"{year}0101", end_de=f"{year + 1}1231",
                      pblntf_detail_ty=pblntf_detail, page_count="100")
        store.save_raw_json(cached, *key)

    if cached.get("status") != "000":
        return None
    items = cached.get("list", [])
    if not items:
        return None

    # 보고서명에 결산기가 붙는다 — "사업보고서 (2025.12)", "분기보고서 (2026.03)".
    # 이 표기로 사업연도를 확정한다. 제출일 범위로만 고르면 사업보고서에서
    # 전년도 보고서를 집어 1년 앞선 날짜가 박히고, 그대로 미래참조가 된다.
    marker = f"({year}.{PERIOD_END[reprt_code]}"
    matched = [it for it in items if marker in (it.get("report_nm") or "").replace(" ", "")]
    if not matched:
        return None

    # 정정 공시가 있으면 최초 공시일을 쓴다. 정정일을 쓰면 그 시점에
    # 몰랐던 정보를 알았던 것처럼 만들거나, 반대로 원본을 못 보게 만든다.
    dates = sorted(it["rcept_dt"] for it in matched if it.get("rcept_dt"))
    return dates[0] if dates else None
