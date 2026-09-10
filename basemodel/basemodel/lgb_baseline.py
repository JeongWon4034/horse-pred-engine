# -*- coding: utf-8 -*-
"""LightGBM 기준선을 별도 프로세스에서 계산해 parquet 로 남긴다.

torch 를 import 하지 않는다 — 두 라이브러리가 각자 OpenMP 런타임을 실어서 한 프로세스에
같이 올리면 libomp 충돌로 죽는다. 설정은 팀 기준선(docs/model/baseline_lgbm.py) 그대로.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# team.py 와 같은 규칙으로 하네스를 찾는다. torch 는 import 하지 않는다 (OpenMP 충돌).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from team_paths import TEAM_DATASET, TEAM_MODEL  # noqa: E402

sys.path.insert(0, str(TEAM_MODEL))
sys.path.insert(0, str(TEAM_DATASET))
import common as C  # noqa: E402

PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[3],
              lambdarank_truncation_level=5, learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1, seed=0, num_threads=4)
N_ROUND = 200


def main() -> None:
    import lightgbm as lgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tr, ev = C.load("train"), C.load(a.split)
    out = pd.DataFrame({"race_id": ev["race_id"].to_numpy(), "hrNo": ev["hrNo"].to_numpy()})
    for name, xp in (("LightGBM 77", False), ("LightGBM 73", True)):
        cols = C.feature_cols(exclude_pop=xp)
        x, y = C.encode(tr, ev, cols)
        ds = lgb.Dataset(x[cols].to_numpy(float), label=x["y_rel"].to_numpy(),
                         group=C.race_groups(x))
        m = lgb.train(PARAMS, ds, num_boost_round=N_ROUND)
        out[name] = m.predict(y[cols].to_numpy(float))
        print(f"  {name}: {C.evaluate(ev, out[name])['top1']:.2f}%")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"  저장 {a.out}")


if __name__ == "__main__":
    main()
