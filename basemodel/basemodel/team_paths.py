# -*- coding: utf-8 -*-
"""팀 하네스(common.py · schema_v2.py) 위치 찾기 — **torch 를 import 하지 않는다.**

`team.py` 와 `lgb_baseline.py` 가 같이 쓴다. lgb_baseline 은 lightgbm 과 torch 의 OpenMP
충돌 때문에 별도 프로세스에서 도는데, 경로 규칙까지 따로 두면 두 곳이 어긋난다.

찾는 순서 — 먼저 걸리는 것을 쓴다.
  1. 환경변수 `TEAM_HARNESS` 가 가리키는 폴더 (그 안에 model/ 과 dataset/)
  2. `pipeline/data/{model,dataset}`  — 이 레포. `pipeline/sync_dataset.sh` 가 채운다
  3. `docs/{model,dataset}`           — 팀 레포(S15P21A304) 안에서 직접 돌릴 때
  4. 환경변수 `TEAM_REPO` 가 가리키는 팀 레포의 docs/
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve()


def candidates() -> list[tuple[Path, Path]]:
    """(model 폴더, dataset 폴더) 후보를 우선순위대로."""
    out: list[tuple[Path, Path]] = []
    if env := os.environ.get("TEAM_HARNESS"):
        p = Path(env).expanduser()
        out.append((p / "model", p / "dataset"))
    for base in _HERE.parents:                      # 파일에서 위로 올라가며 찾는다
        out.append((base / "pipeline" / "data" / "model", base / "pipeline" / "data" / "dataset"))
        out.append((base / "docs" / "model", base / "docs" / "dataset"))
    if env := os.environ.get("TEAM_REPO"):
        p = Path(env).expanduser() / "docs"
        out.append((p / "model", p / "dataset"))
    return out


def resolve() -> tuple[Path, Path]:
    """(TEAM_MODEL, TEAM_DATASET). 없으면 무엇을 해야 하는지 알려주고 멈춘다 —
    조용히 다른 데이터를 읽는 것보다 낫다."""
    for model_dir, dataset_dir in candidates():
        if (model_dir / "common.py").exists() and (dataset_dir / "schema_v2.py").exists():
            return model_dir, dataset_dir
    raise FileNotFoundError(
        "팀 하네스를 찾지 못했다 (common.py + schema_v2.py).\n"
        "  이 레포에서 돌린다면:  cd pipeline && bash sync_dataset.sh\n"
        "  사본이 다른 곳에 있다면:  TEAM_HARNESS=/경로/data  또는  TEAM_REPO=/경로/S15P21A304\n"
        "  찾아본 곳(앞 6개): " + ", ".join(str(m.parent) for m, _ in candidates()[:6]))


TEAM_MODEL, TEAM_DATASET = resolve()
ARTIFACTS = _HERE.parents[1] / "artifacts"
