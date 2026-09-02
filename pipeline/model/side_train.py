"""사이드 실험 — S3 모델 구조(GRU 크기·dropout·임베딩·헤드) 탐색.

train.py 를 고치지 않고 build() 만 바꿔 끼운다. 결과 예측·가중치는 side_ 접두사로 저장돼 본 실험과 섞이지 않는다.

    PYTHONUTF8=1 uv run python -m model.side_train --gru 128 --dropout 0.3 --seed 1
"""
from __future__ import annotations

import argparse

from . import train as T
from .history import FEATS
from .models import HistoryRanker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gru", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--hist-len", type=int, default=20)
    ap.add_argument("--seed", type=int, default=T.SEED)
    ap.add_argument("--market", action="store_true")
    ap.add_argument("--name", required=True, help="이 구성의 이름 (파일 접미사)")
    a = ap.parse_args()

    def build(kind, enc, vocabs, dev):
        return HistoryRanker(enc.dim, [v.size for v in vocabs.values()], len(FEATS),
                             emb=a.emb, gru=a.gru, hidden=a.hidden, dropout=a.dropout).to(dev)

    T.build = build                      # train.run 이 이 build 를 쓴다
    print(f"[side] {a.name}: gru {a.gru} dropout {a.dropout} emb {a.emb} hidden {a.hidden} "
          f"lr {a.lr} wd {a.wd} bs {a.bs} topk {a.topk} min_count {a.min_count} L {a.hist_len} seed {a.seed}")
    T.run("history", a.market, a.epochs, a.bs, a.lr, a.wd, a.patience, a.topk, a.seed, a.min_count,
          tag_suffix=f"_side_{a.name}_s{a.seed}", hist_len=a.hist_len)


if __name__ == "__main__":
    main()
