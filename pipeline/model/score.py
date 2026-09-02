"""저장된 모델로 다른 분할(game)의 점수를 뽑는다 — 재학습 없이.

    PYTHONUTF8=1 uv run python -m model.score --split game --ckpt history_73_L20_s1 --hist-len 20
    PYTHONUTF8=1 uv run python -m model.score --split game --lgb            # LightGBM 73/77 도 game 점수
    → experiments/pred/{ckpt}_game.parquet  (race_id · hrNo · score),  model.backtest --pred 로 ROI

game 은 학습·검증에 절대 쓰지 않는다. 여기서는 오직 점수를 뽑아 ROI 백테스트에 넘기기만 한다.
전처리(표준화·원핫·어휘·이력 통계)는 train 에서 다시 fit 한다 — 결정적이므로 학습 때와 같은 값이 나온다.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .categorical import EMBED_COLS, encode_cats, fit_vocabs
from .data import feature_cols, fit_encoder, flatten_scores, load, to_races
from .evaluate import save_pred
from .history import fit_stats, history_for, load_pool
from .team import C, S, TEAM_DATASET
from .train import RUNS, build, predict

VALID_SPLITS = ("game", "valid")      # test 는 여기서도 열지 않는다


def load_split(split: str) -> pd.DataFrame:
    if split == "valid":
        return load("valid")
    if split == "game":
        df = pd.read_parquet(TEAM_DATASET / "model" / "game.parquet")
        codes = pd.factorize(df["race_id"], sort=False)[0]
        assert not (np.diff(codes) < 0).any(), "game race_id 정렬 깨짐"
        return df
    raise ValueError(f"split 은 {VALID_SPLITS} 중 하나. test 는 열지 않는다")


def score_dl(ckpt: str, split: str, hist_len: int) -> Path:
    ck = torch.load(RUNS / f"{ckpt}.pt", weights_only=False)
    kind, market = ck["kind"], ck["market"]
    m = re.search(r"_L(\d+)", ckpt)                      # 태그에 이력 길이가 있으면 그것을 쓴다
    hist_len = ck.get("hist_len", int(m.group(1)) if m else hist_len)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr_df = load("train")
    df = load_split(split)
    cols = feature_cols(exclude_pop=not market)
    enc = fit_encoder(tr_df, cols)
    assert enc.names == ck["names"], "인코더가 학습 때와 다르다 — 데이터 사본이 바뀌었나"
    vocabs = fit_vocabs(tr_df, EMBED_COLS) if kind != "linear" else {}
    for c, v in vocabs.items():
        assert v.index == ck["vocabs"][c], f"{c} 어휘가 학습 때와 다르다"

    extra = {}
    if kind in ("embed", "history", "attn"):
        extra["cat"] = encode_cats(df, vocabs)
    if kind in ("history", "attn"):
        pool = load_pool()
        stats = fit_stats(pool, int(tr_df["rcDate"].max()))
        extra["hist"], extra["hist_len"] = history_for(df, split, pool, stats, L=hist_len)
    races = to_races(df, enc, extra)

    model = build(kind, enc, vocabs, dev)
    model.load_state_dict(ck["state"])
    T = {k: torch.as_tensor(v, device=dev) for k, v in races.extra.items()}
    scores = flatten_scores(predict(model, races, T, dev), races)
    out = save_pred(df, scores, f"{ckpt}_{split}")
    print(f"[{ckpt}] {split} {df['race_id'].nunique():,}경주 {len(df):,}두 → {out.name}")
    return out


def score_lgb(split: str) -> list[Path]:
    import baseline_lgbm as B  # team.py 가 경로를 잡아 둔 뒤라 import 가능
    tr_df, df = load("train"), load_split(split)
    outs = []
    for n_feat, exclude_pop in ((77, False), (73, True)):
        _, s, cols = B.train_and_score(tr_df, df, exclude_pop)
        outs.append(save_pred(df, s, f"lgb_{n_feat}_{split}"))
        print(f"[lgb_{n_feat}] {split} → {outs[-1].name}")
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="game", choices=VALID_SPLITS)
    ap.add_argument("--ckpt", nargs="*", default=[], help="experiments/runs 의 .pt 이름 (확장자 없이)")
    ap.add_argument("--hist-len", type=int, default=10, help="체크포인트에 없을 때 쓸 이력 길이")
    ap.add_argument("--lgb", action="store_true", help="LightGBM 73/77 도 같은 분할에 점수")
    a = ap.parse_args()
    torch.manual_seed(0)
    if a.lgb:
        score_lgb(a.split)
    for c in a.ckpt:
        score_dl(c, a.split, a.hist_len)


if __name__ == "__main__":
    main()
