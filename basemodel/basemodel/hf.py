# -*- coding: utf-8 -*-
"""허깅페이스 배포 — 가중치·설정·전처리·모델카드를 한 폴더로 묶는다.

    uv run python -m basemodel.hf --ckpt artifacts/runs/axis_77_s20260901.pt \
        --out artifacts/hf/latte-axis-ranker
    uv run python -m basemodel.hf --ckpt ... --push <user>/latte-axis-ranker

`from_pretrained` 로 다시 불러오려면 가중치만으로는 부족하다. 이 모델은 표 데이터를
받으므로 **전처리(중앙값·평균·표준편차·원핫 어휘·범주형 사전)** 가 같이 가야 하고,
그게 없으면 남이 받아서 쓸 수 없다. 그래서 preprocessor.json 을 같이 낸다.

safetensors 로 저장한다(pickle 이 아니라서 남의 가중치를 받아도 코드가 실행되지 않는다).
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from . import config as cfg
from . import history as H
from .team import ARTIFACTS

CARD = """---
license: mit
language: [ko]
tags: [horse-racing, learning-to-rank, plackett-luce, tabular, pytorch]
library_name: pytorch
---

# {repo} — 경마 예측 6축 랭커 (AxisRanker)

경마 예측을 분류가 아니라 **랭킹** 문제로 푼다. 경주 하나(최대 16두)를 통째로 받아
말마다 점수를 내고, 경주 안에서 softmax 를 걸어 1착 확률을 만든다.

핵심은 점수가 **6갈래로 나뉘어** 나온다는 것이다. 사용자가 여섯 축의 비율을
실시간으로 바꿔도 **재학습이 필요 없다** — 축 점수는 미리 확정되고, 화면은 곱셈·덧셈만 한다.

```
최종 점수 = m · s_시장 + Σ_k  w_k · s_k        Σ w_k = 100 (사용자), m = 프리셋 상수
1착 확률  = softmax(최종 점수 × T)             경주 안에서
```

## 축 여섯 개

| 키 | 이름 | 무엇을 보는가 |
|---|---|---|
{axis_rows}

`MARKET`(시장·배당)은 **슬라이더가 아니다.** 프리셋마다 고정 계수 `m` 으로 들어가고,
`m < 0` 이면 시장이 저평가한 말을 고른다(역배형).

## 베이스 모델 {n_preset}종

{preset_rows}

가중치는 손으로 고른 값이 아니라 **학습 구간에서 목적함수로 맞춘 값**이다.

## 성능 (valid {n_races:,}경주 · {n_feat}피처 · top-1 표준오차 ±{se:.2f}%p)

| 모델 | top-1 | top-3 |
|---|---:|---:|
{perf_rows}

표준오차의 두 배({se2:.1f}%p)를 넘어야 우열로 읽는다.

## 구조

| 부품 | 역할 | 소속 |
|---|---|---|
| `nn.Linear` × 2 (trunk) | 경주 상황(거리·등급·두수·날씨·주로) 공유 표현 | 전 축 공유 |
| `nn.Embedding` × 3 | 기수·조교사·부마 ID → {emb}차원 | 기수 / 실력 축 |
| `nn.GRU` | 말별 직전 {hist_len}출전을 순서대로 읽음 | 축마다 다른 선형 읽기 |
| MLP × {n_axes}(+1) | 축 점수 → 경주 내 z-score | 축 하나씩 |

- 파라미터 {n_param:,}개. 사전학습 가중치 없음 — 이 데이터로만 0 에서 학습했다.
- 손실: **Plackett–Luce** listwise (1~3착까지 전개). 조건부 로지스틱 회귀 = ListNet 과 같은 뼈대.
- 학습 중 가중치 `w` 를 **디리클레(α={alpha})**, 시장 계수 `m` 을 균등분포에서 매 배치 새로 뽑는다.
  그래야 사용자가 극단 조합을 넣어도 무너지지 않는다.
- 축 간 상관 벌점(λ={decorr})으로 축이 서로 다른 말을 가리키게 눌렀다.

## 쓰는 법

```python
import json, torch
from safetensors.torch import load_file

cfg_ = json.load(open("config.json"))
state = load_file("model.safetensors")
# 구조 정의는 basemodel/model.py 의 AxisRanker 를 그대로 쓴다
```

전처리는 `preprocessor.json` 에 있다 — 수치 피처의 중앙값·평균·표준편차, 원핫 어휘,
범주형 사전, 이력 표준화 상수. 이게 없으면 입력을 만들 수 없다.

## 데이터 · 한계

- 한국마사회 공공 API, {data_period}. 학습 {n_train:,}경주.
- 시간순 분할. `game` 3,700경주는 학습·검증·튜닝 어디에도 쓰지 않았다.
- 배당(`MARKET` 축)은 **확정배당**이라 경주 후 값이다. 과거 경주 리플레이에서만 쓸 수 있고
  주말 실시간 예측에는 못 쓴다(발주 전 배당 미제공).
- ROI 는 공제율 20%(실측 환급률 80%) 아래에서 어떤 조합도 음수다. 역배형의 목표는
  수익이 아니라 **고배당 적중**이다.

## 출처

- 손실·이력 GRU·평가 하네스는 팀 실험 레포(이정원, `JeongWon4034/horse-pred-engine`)의
  S1~S4 단계에서 가져왔다. 6축 타워·시장 축·탈상관 벌점·프리셋 적합이 이 레포에서 더한 부분이다.
- Benter(1994) 조건부 로지스틱 회귀 · Plackett(1975)/Luce(1959) 순위 확률 모델.
"""


def export(ckpt: Path, out: Path, meta_path: Path | None = None,
           repo: str = "latte-axis-ranker") -> Path:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    with open(Path(str(ckpt).replace(".pt", ".enc.pkl")), "rb") as f:
        side = pickle.load(f)
    enc = side["encoder"]
    out.mkdir(parents=True, exist_ok=True)

    # ── 가중치 (safetensors) ────────────────────────────────────────
    state = {k: v.contiguous() for k, v in blob["state"].items()}
    save_file(state, str(out / "model.safetensors"))

    # ── config.json — 구조를 되살리는 데 필요한 전부 ─────────────────
    config = {
        "architectures": ["AxisRanker"],
        "model_type": "axis-ranker",
        "axes": blob["axes"],
        "market_axis": cfg.MARKET if blob["market"] else None,
        "score_columns": blob["axes"] + ([cfg.MARKET] if blob["market"] else []),
        "slices": {k: np.asarray(v).tolist() for k, v in blob["slices"].items()},
        "vocab_sizes": blob["vocab_sizes"],
        "k_hist": blob["k_hist"],
        "hist_len": cfg.HIST_LEN,
        "emb_dim": cfg.EMB_DIM,
        "gru_dim": cfg.GRU_DIM,
        "tower_hidden": cfg.TOWER_HIDDEN,
        "trunk_hidden": cfg.TRUNK_HIDDEN,
        "dropout": cfg.DROPOUT,
        "max_field": cfg.MAX_FIELD,
        "input_dim": enc.dim,
        "n_features": 77 if blob["market"] else 73,
        "dirichlet_alpha": cfg.DIRICHLET_ALPHA,
        "decorr_lambda": blob.get("decorr", cfg.DECORR_LAMBDA),
        "seed": blob["seed"],
    }
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), "utf-8")

    # ── preprocessor.json — 이게 없으면 남이 못 쓴다 ─────────────────
    pre = {
        "feature_names": enc.names,
        "axis_of": enc.axis_of,
        "numeric": enc.numeric,
        "median": {k: float(v) for k, v in enc.median.items()},
        "mean": {k: float(v) for k, v in enc.mean.items()},
        "std": {k: float(v) for k, v in enc.std.items()},
        "onehot": {k: list(map(str, v)) for k, v in enc.onehot.items()},
        "na_flag": enc.na_flag,
        "clip": 8.0,
        "categorical_vocab": {c: v.index for c, v in side["vocabs"].items()},
        "categorical_special": {"pad": 0, "unk": 1, "rare": 2},
        "history": {"feats": H.FEATS, "len": cfg.HIST_LEN, "stats": side["hist_stats"]},
    }
    (out / "preprocessor.json").write_text(json.dumps(pre, ensure_ascii=False, indent=2), "utf-8")

    # 구조 정의 파일을 같이 넣는다 — 카드에서 model.py 를 가리키므로 실물이 있어야 한다
    src = Path(__file__).parent
    for f in ("model.py", "losses.py", "config.py"):
        shutil.copy(src / f, out / f)

    write_card(out, config, meta_path, repo, n_param=sum(v.numel() for v in state.values()))
    return out


def write_card(out: Path, config: dict, meta_path: Path | None, repo: str, n_param: int) -> None:
    meta = json.loads(Path(meta_path).read_text("utf-8")) if meta_path and Path(meta_path).exists() else {}
    axis_desc = {
        "CONDITION": "최근 착순 흐름·휴양·조교 — 지금 상태가 좋은가",
        "SPEED": "주파기록에서 나온 속도와 유전적 스피드",
        "RUNNING": "각질·초반 위치·막판 각력 — 어떻게 달리는가",
        "JOCKEY": "기수·조교사 성적과 말-기수 궁합",
        "ENVIRONMENT": "이번 거리·주로·날씨에 맞는가",
        "ABILITY": "마사회 레이팅과 혈통 육종가 — 공인된 실력",
    }
    axis_rows = "\n".join(
        f"| `{ax}` | {cfg.AXIS_KO[ax]} | {axis_desc.get(ax, '')} |" for ax in config["axes"])

    pres = meta.get("presets", {})
    if pres:
        preset_rows = "| 키 | 이름 | 목적 | 시작 가중치 | 시장 계수 m |\n|---|---|---|---|---:|\n" + "\n".join(
            f"| `{k}` | {v['label']} | {v['desc']} | "
            + "·".join(f"{cfg.AXIS_KO[a]} {w}" for a, w in v["groupWeights"].items() if w)
            + f" | {v['market']:+.2f} |" for k, v in pres.items())
    else:
        preset_rows = "_meta.json 이 없어 비어 있다 — `basemodel.export` 를 먼저 돌릴 것._"

    val = meta.get("validation", {})
    se = val.get("equal_weight", {}).get("se", 0.85)
    perf = [("시장 (인기 1위마, 참고)", None, None)]
    rows = []
    eqm = val.get("equal_weight", {})
    if eqm:
        rows.append(f"| **AxisRanker (균등 가중, 유저 기본값)** | {eqm.get('top1', 0):.1f} | {eqm.get('top3', 0):.1f} |")
    for k, v in val.get("presets", {}).items():
        rows.append(f"| {pres.get(k, {}).get('label', k)} | {v.get('top1', 0):.1f} | {v.get('top3', 0):.1f} |")
    perf_rows = "\n".join(rows) or "| — | — | — |"

    card = CARD.format(
        repo=repo, axis_rows=axis_rows, preset_rows=preset_rows,
        n_preset=len(pres) or len(cfg.PRESETS), perf_rows=perf_rows,
        n_races=val.get("n_races", 0), n_feat=config["n_features"],
        se=se, se2=2 * se, emb=config["emb_dim"], hist_len=config["hist_len"],
        n_axes=len(config["axes"]), n_param=n_param,
        alpha=config["dirichlet_alpha"], decorr=config["decorr_lambda"],
        data_period="2010~2026", n_train=36586,
    )
    (out / "README.md").write_text(card, "utf-8")


def load_pretrained(folder: str | Path):
    """허깅페이스 폴더에서 모델과 전처리를 되살린다.

        model, pre = load_pretrained("artifacts/hf/latte-axis-ranker")

    `basemodel` 패키지 없이도 되게 하려고 폴더 안에 model.py·config.py 를 같이 넣는다.
    여기서는 패키지가 있는 환경을 가정하고 구조만 재구성한다.
    """
    from safetensors.torch import load_file
    from .model import AxisRanker

    folder = Path(folder)
    conf = json.loads((folder / "config.json").read_text("utf-8"))
    pre = json.loads((folder / "preprocessor.json").read_text("utf-8"))
    slices = {k: np.asarray(v, dtype=np.int64) for k, v in conf["slices"].items()}
    model = AxisRanker(slices, conf["vocab_sizes"], k_hist=conf["k_hist"],
                       emb=conf["emb_dim"], gru=conf["gru_dim"],
                       tower_hidden=conf["tower_hidden"], trunk_hidden=conf["trunk_hidden"],
                       dropout=conf["dropout"])
    model.load_state_dict(load_file(str(folder / "model.safetensors")))
    model.eval()
    return model, pre


def _verify(ckpt: Path, folder: Path) -> None:
    """폴더에서 되살린 모델이 원본과 **같은 점수**를 내는지 확인한다.

    가중치만 맞아도 소용없다 — 전처리 상수가 틀리면 입력이 달라지고 점수도 달라진다.
    그래서 실제 데이터(valid) 한 조각을 통과시켜 원본과 비교한다.
    """
    from .serve import Scorer
    from .team import C

    df = C.load("valid").head(2000)
    ref = Scorer(ckpt, device="cpu").axis_rows(df, "valid")

    model, pre = load_pretrained(folder)
    # 전처리를 preprocessor.json 만 보고 다시 만든다 — 원본 pickle 을 쓰지 않는다
    import pandas as pd
    from . import data as D
    enc = D.Encoder(
        numeric=pre["numeric"],
        onehot={k: v for k, v in pre["onehot"].items()},
        median=pd.Series(pre["median"]), mean=pd.Series(pre["mean"]), std=pd.Series(pre["std"]),
        na_flag=pre["na_flag"], axis_of=pre["axis_of"], names=pre["feature_names"])
    from . import categorical as cm
    vocabs = {c: cm.Vocab(c, {k: int(v) for k, v in idx.items()}, 0)
              for c, idx in pre["categorical_vocab"].items()}
    from . import history as H
    hist, hlen = H.history_for(df, "valid", stats=H.HistStats(**pre["history"]["stats"]),
                               L=pre["history"]["len"], verbose=False)
    races = D.to_races(df, enc, {"cat": cm.encode_cats(df, vocabs), "hist": hist, "hist_len": hlen})
    with torch.no_grad():
        s = model.axis_scores(torch.as_tensor(races.x), torch.as_tensor(races.mask),
                              torch.as_tensor(races.extra["cat"]),
                              torch.as_tensor(races.extra["hist"]),
                              torch.as_tensor(races.extra["hist_len"])).numpy()
    got = s[races.row_race, races.row_slot]
    diff = float(np.abs(got - ref).max())
    print(f"\n[검증] valid {len(df):,}행 재현 — 최대 점수 차 {diff:.2e}  "
          f"{'통과' if diff < 1e-4 else '★ 불일치'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=str(ARTIFACTS / "hf" / "latte-axis-ranker"))
    ap.add_argument("--meta", default=str(ARTIFACTS / "export" / "meta.json"))
    ap.add_argument("--push", default=None, help="허브 repo id (예: user/latte-axis-ranker)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="폴더에서 다시 불러와 원본 체크포인트와 점수가 같은지 확인")
    a = ap.parse_args()

    out = Path(a.out)
    export(Path(a.ckpt), out, Path(a.meta), repo=a.push or out.name)
    total = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(f"[허깅페이스 폴더] {out}  ({total / 1e6:.2f}MB)")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:<24}{f.stat().st_size / 1024:>9.1f}KB")

    if a.verify:
        _verify(Path(a.ckpt), out)

    if a.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.push, private=a.private, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=a.push)
        print(f"[업로드] https://huggingface.co/{a.push}")
    else:
        print("\n업로드하려면:  --push <user>/latte-axis-ranker   (huggingface-cli login 먼저)")


if __name__ == "__main__":
    main()
