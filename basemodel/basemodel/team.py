# -*- coding: utf-8 -*-
"""팀 공용 채점기(`docs/model/common.py`) 진입점.

데이터를 직접 읽지 않는다. 읽기·피처 선택·정렬 검사·적중률 계산은 전부 팀 채점기가
한다. 그래야 LightGBM(도연)·S1~S5(정원)·이 모델이 **같은 행, 같은 피처, 같은 자**로
비교된다.

    from .team import C, S
    tr = C.load("train")                    # clean + usable + 정렬 검사
    cols = C.feature_cols(exclude_pop=True) # 73개 (인기도 제외, 주말 실시간 조건)
    print(C.report({"내 모델": scores}, va))

## 하네스를 어디서 찾나

이 코드는 두 레포에서 다 돌아간다. 위에서부터 찾아서 먼저 걸리는 것을 쓴다.

  1. 환경변수 `TEAM_HARNESS` 가 가리키는 폴더 (그 안에 model/ 과 dataset/)
  2. `pipeline/data/{model,dataset}`  — 이 레포. `pipeline/sync_dataset.sh` 가 채운다
  3. `docs/{model,dataset}`           — 팀 레포(S15P21A304) 안에서 직접 돌릴 때
  4. 환경변수 `TEAM_REPO` 가 가리키는 팀 레포의 docs/

없으면 무엇을 해야 하는지 알려주고 멈춘다 — 조용히 다른 데이터를 읽는 것보다 낫다.
"""
from __future__ import annotations

import sys

from .team_paths import ARTIFACTS, TEAM_DATASET, TEAM_MODEL  # noqa: F401

for _p in (TEAM_MODEL, TEAM_DATASET):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import common as C      # noqa: E402
import schema_v2 as S   # noqa: E402

__all__ = ["C", "S", "TEAM_MODEL", "TEAM_DATASET", "ARTIFACTS"]
