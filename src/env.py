"""환경변수 로딩.

상위 프로젝트(주식시장/.env)와 screener/.env를 순서대로 읽는다.
screener/.env가 뒤에 오므로 같은 키가 있으면 이쪽이 이긴다.
값은 절대 로그에 찍지 않는다.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import config

_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_loaded = False


def _read(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        os.environ[key] = val
        n += 1
    return n


def load(force: bool = False) -> None:
    global _loaded
    if _loaded and not force:
        return
    _read(config.ROOT.parent / ".env")   # 주식시장/.env  (KRX_ID, KRX_PW 등)
    _read(config.ROOT / ".env")          # screener/.env  (DART_API_KEY)
    config.DART_KEY = os.environ.get("DART_API_KEY", "")
    # config 를 임포트할 때는 .env 를 아직 안 읽었으므로 여기서 다시 반영한다
    if os.environ.get("VAULT_ROOT"):
        config.VAULT_ROOT = os.environ["VAULT_ROOT"]
    _loaded = True


def status() -> dict[str, bool]:
    """어떤 키가 들어왔는지만 boolean으로. 값은 반환하지 않는다."""
    load()
    return {k: bool(os.environ.get(k)) for k in
            ("KRX_ID", "KRX_PW", "DART_API_KEY")}
