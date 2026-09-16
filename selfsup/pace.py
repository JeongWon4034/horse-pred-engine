"""P6 — 과거 경주의 페이스 곡선을 이력 텐서에 붙인다 (b 방향).

정원 S3 의 이력 항목 10개는 과거 경주 하나를 요약값(착순·스피드·거리…)으로 넣는다.
그 경주 **안에서 어떻게 달렸는지**(초반 몇 위 → 중반 → 막판 → 착순)는 F5 가 평균으로 뭉갠 채로만 있다.
여기서는 원장 CSV 의 구간 통과순위를 과거 경주마다 그대로 붙인다 — 정원 이력 10열 + 페이스 5열.

  1) python -m selfsup.pace build     원장 → runs/pace.parquet  (건모 build_v2.add_sections 그대로 사용)
  2) hybrid.py --pace                 이력 텐서에 페이스 열을 붙여 학습 (대조군은 --pace 없이)

원장은 팀 레포 docs/dataset/data/raw/ledger_2010_2026.csv (backfill_ledger.py 산출, git 에 없음).
페이스 열 (과거 경주 1건당):
  early · mid · late  = 구간 통과순위 (0=선두 … 1=꼴찌, 두수 정규화)   ← 건모 _pos_*
  gain                = early − 최종 착순 위치 (양수 = 추입해서 올라옴)
  na                  = 구간기록 없음 (제주 전 경주·결측)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D

TEAM = Path(r"C:\git\S15P21A304")
DATASET_DIR = TEAM / "docs" / "dataset"
PACE = D.RUNS / "pace.parquet"
FEATS = ["p_early", "p_mid", "p_late", "p_gain", "p_na"]


LEDGER_API = D.PIPE / "data" / "raw" / "ledger_api.csv"      # 정원 ingest.ledger 산출 (API214_1)


def build() -> Path:
    """원장 → (race_id, hrNo, _pos_early, _pos_mid, _pos_late, ordpct). 건모 코드로 구간 처리.

    원장은 정원 `ingest.ledger` 의 ledger_api.csv 를 쓴다 (팀 API4_3 는 우리 키로 403).
    build_live.sh 와 같은 변환 하나 — 경마장 표기 '부경' → '부산경남'."""
    # 정원 build_live.sh 가 만든 폴더(pipeline/data/build)를 그대로 쓴다 — 팀 빌더가 기대하는
    # data/raw/ledger_2010_2026.csv · dataset/v2/game_holdout.json 배치가 거기 이미 되어 있다
    build_dir = D.PIPE / "data" / "build"
    if not (build_dir / "build_v2.py").exists():
        raise FileNotFoundError(f"{build_dir} 없음 → cd pipeline && bash build_live.sh")
    cwd = os.getcwd(); os.chdir(build_dir)
    try:
        sys.path.insert(0, str(build_dir))
        import build_v2 as B
        df = B.load_ledger()
        df = B.assign_split(df)
        df = B.add_sections(df)
    finally:
        os.chdir(cwd)
    out = pd.DataFrame({
        "race_id": df["race_id"].to_numpy(), "hrNo": df["hrNo"].astype(str).to_numpy(),
        "early": df["_pos_early"].to_numpy(float), "mid": df["_pos_mid"].to_numpy(float),
        "late": df["_pos_late"].to_numpy(float),
        "fin": ((df["ord"].to_numpy(float) - 1) / np.maximum(df["dusu"].to_numpy(float) - 1, 1)),
    })
    out.to_parquet(PACE, index=False)
    cov = out[["early", "mid", "late"]].notna().mean()
    print(f"→ {PACE}  행 {len(out):,}  구간기록 충전 early {cov['early']:.2f} mid {cov['mid']:.2f} late {cov['late']:.2f}")
    return PACE


def history_pace_for(target: pd.DataFrame, L: int) -> np.ndarray:
    """정원 history.build 와 같은 정렬·인덱스로 과거 L 출전의 페이스 5열을 만든다. [N, L, 5]."""
    from model.history import load_pool
    if not PACE.exists():
        raise FileNotFoundError(f"{PACE} 없음 → python -m selfsup.pace build")
    pace = pd.read_parquet(PACE)
    pool = load_pool()                                             # train ∪ valid, game·test 제외 assert 포함
    pool = pool.merge(pace, on=["race_id", "hrNo"], how="left")    # 없는 경주는 NaN
    assert len(pool) == len(pool.drop_duplicates(["race_id", "hrNo"])), "원장 조인이 행을 늘렸다"

    hr_all = pd.concat([pool["hrNo"], target["hrNo"].astype(str)], ignore_index=True)
    codes, _ = pd.factorize(hr_all, sort=True)
    pool_code, tgt_code = codes[:len(pool)], codes[len(pool):]
    pool_key = pool_code.astype(np.int64) * 10**8 + pool["rcDate"].to_numpy(np.int64)
    assert (np.diff(pool_key) >= 0).all()
    tgt_date = target["rcDate"].to_numpy(np.int64)
    tgt_key = tgt_code.astype(np.int64) * 10**8 + tgt_date
    end = np.searchsorted(pool_key, tgt_key, side="left")
    start = np.searchsorted(pool_key, tgt_code.astype(np.int64) * 10**8, side="left")
    n_hist = np.minimum(L, end - start)
    pos = np.arange(L)[None, :]
    valid = pos < n_hist[:, None]
    idx = (end[:, None] - n_hist[:, None] + pos).clip(0, len(pool) - 1)
    assert (pool["rcDate"].to_numpy()[idx][valid] < np.broadcast_to(tgt_date[:, None], idx.shape)[valid]).all()

    e = pool["early"].to_numpy(float)[idx]; m = pool["mid"].to_numpy(float)[idx]
    l = pool["late"].to_numpy(float)[idx]; f = pool["fin"].to_numpy(float)[idx]
    na = np.isnan(e) & np.isnan(m) & np.isnan(l)
    feats = np.stack([
        np.nan_to_num(e, nan=0.5), np.nan_to_num(m, nan=0.5), np.nan_to_num(l, nan=0.5),
        np.nan_to_num(e - f, nan=0.0), na.astype(float),
    ], axis=-1).astype(np.float32)
    feats[~valid] = 0.0
    cov = 1 - na[valid].mean() if valid.any() else 0.0
    print(f"[페이스] 유효 이력 슬롯 중 구간기록 있음 {cov*100:.1f}%")
    return feats


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build()
    else:
        print(__doc__)
