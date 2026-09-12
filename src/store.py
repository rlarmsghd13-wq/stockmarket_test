"""저장소 — parquet 기반.

조회 패턴이 (ticker, asof_date) 범위 스캔이라 parquet + pyarrow로 충분하다.
DB 서버를 띄울 이유가 없다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config


def _path(name: str) -> Path:
    return config.CURATED / f"{name}.parquet"


def save(df: pd.DataFrame, name: str) -> Path:
    p = _path(name)
    df.to_parquet(p, index=False)
    return p


def load(name: str) -> pd.DataFrame:
    p = _path(name)
    if not p.exists():
        raise FileNotFoundError(f"{p} 가 없습니다. 앞 단계 스크립트를 먼저 실행하세요.")
    return pd.read_parquet(p)


def exists(name: str) -> bool:
    return _path(name).exists()


def save_raw_json(payload: dict, *parts: str) -> Path:
    """DART 원본 응답 보존. 파싱 로직을 고쳐도 재수집하지 않기 위해서다."""
    p = config.RAW.joinpath(*parts).with_suffix(".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return p


def load_raw_json(*parts: str) -> dict | None:
    p = config.RAW.joinpath(*parts).with_suffix(".json")
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 시점 잠금 조회 — 이 프로젝트에서 가장 중요한 함수
# ---------------------------------------------------------------------------
def asof(df: pd.DataFrame, asof_date: str, *, date_col: str = "rcept_dt",
         key: str = "ticker") -> pd.DataFrame:
    """`asof_date` 시점에 실제로 알 수 있었던 최신 레코드만 종목별로 하나씩.

    공시 접수일(rcept_dt)이 asof_date 이후인 행은 결과에 절대 포함되지 않는다.
    백테스트의 미래참조(look-ahead)를 막는 유일한 장치이므로
    재무 데이터 조회는 반드시 이 함수를 거친다.
    """
    if date_col not in df.columns:
        raise KeyError(f"{date_col} 컬럼이 없습니다. 시점 잠금이 불가능합니다.")
    asof_ts = pd.Timestamp(asof_date)
    visible = df[pd.to_datetime(df[date_col]) <= asof_ts]
    if visible.empty:
        return visible
    idx = visible.groupby(key)[date_col].idxmax()
    return visible.loc[idx].reset_index(drop=True)
