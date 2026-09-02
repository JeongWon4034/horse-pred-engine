"""팀 비교 하네스(docs/model/common.py) 진입점.

딥러닝과 LightGBM 을 같은 자로 재려면 데이터 로드·피처 선택·평가가 전부 같아야 한다.
그 셋은 팀 레포의 common.py 가 고정하고 있으므로, 이 레포의 모델 코드는
데이터를 직접 읽지 않고 반드시 여기를 거친다.

    from model.team import C
    tr, va = C.load("train"), C.load("valid")      # clean + usable + 정렬 검사
    cols   = C.feature_cols(exclude_pop=True)      # 73개 (인기도 제외, 실시간 조건)
    tr, va = C.encode(tr, va, cols)
    print(C.report({"내 모델": scores}, va))       # 시장·무작위 기준선 포함 표

사본은 pipeline/data/model/ 에 있고 git 에 올리지 않는다.
없거나 오래됐으면  cd pipeline && bash sync_dataset.sh  (팀 origin/develop 의 docs/model).
"""
from __future__ import annotations

import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
TEAM_MODEL = DATA / "model"
TEAM_DATASET = DATA / "dataset"

_missing = [p for p in (TEAM_MODEL / "common.py", TEAM_DATASET / "schema_v2.py") if not p.exists()]
if _missing:
    raise FileNotFoundError(
        "팀 하네스 사본이 없다: " + ", ".join(str(p) for p in _missing)
        + "\n  → cd pipeline && bash sync_dataset.sh   (팀 레포 origin/develop 의 docs/dataset, docs/model)"
    )

if str(TEAM_MODEL) not in sys.path:
    sys.path.insert(0, str(TEAM_MODEL))

import common as C  # noqa: E402  (common 이 내부에서 ../dataset/schema_v2 를 잡는다)
import schema_v2 as S  # noqa: E402

__all__ = ["C", "S", "DATA", "TEAM_MODEL", "TEAM_DATASET"]
