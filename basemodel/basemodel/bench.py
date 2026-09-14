# -*- coding: utf-8 -*-
"""서비스 성능 측정 — "15초 안에 되나"에 숫자로 답한다.

    uv run python -m basemodel.bench --ckpt artifacts/runs/axis_77_s20260901_final.pt

세 가지를 잰다.
  1. 사전 계산   게임풀 3,700경주 전체 점수. 배포 때 1회. (기능명세서 AI-05)
  2. 실시간 추론 새 출전표 한 경주. 주말 실경기용. 사전 계산이 불가능한 경우.
  3. 슬라이더    유저가 비율을 바꿀 때. 모델이 돌지 않고 곱셈·덧셈만 한다. (AI-06)

동시 사용자 10명은 3번만 병렬로 일어난다. 1·2번은 유저 요청 경로에 없다.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .serve import Scorer, combine, load_game
from .team import C


def _timeit(fn, repeat: int = 5) -> tuple[float, float]:
    """(중앙값 초, 최소 초). 첫 회는 워밍업으로 버린다."""
    fn()
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), float(np.min(ts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--users", type=int, default=10)
    a = ap.parse_args()

    sc = Scorer(a.ckpt, device=a.device)
    A = sc.n_axes
    gm = load_game()
    n_races = gm["race_id"].nunique()
    print(f"[환경] device={a.device}  축 {A}(+{'1' if sc.market else '0'})  "
          f"게임풀 {n_races:,}경주 {len(gm):,}두\n")

    # 1. 사전 계산 — 배포 때 1회
    t0 = time.perf_counter()
    gm_ax = sc.axis_rows(gm, "game")
    dt = time.perf_counter() - t0
    print(f"1) 사전 계산 (배포 1회, 유저 경로 아님)")
    print(f"   게임풀 {n_races:,}경주 전체        {dt:6.2f}s   ({dt / n_races * 1000:.2f}ms/경주)")

    # 2. 실시간 추론 — 새 출전표 한 경주 / 하루치
    one = gm[gm["race_id"] == gm["race_id"].iloc[0]]
    med, best = _timeit(lambda: sc.axis_rows(one, "game"), repeat=10)
    day = gm[gm["rcDate"] == gm["rcDate"].iloc[0]]
    med_d, _ = _timeit(lambda: sc.axis_rows(day, "game"), repeat=3)
    print(f"\n2) 실시간 추론 (사전 계산이 없을 때)")
    print(f"   1경주 {len(one):>2}두                  {med * 1000:6.1f}ms  (최소 {best * 1000:.1f}ms)")
    print(f"   하루치 {day['race_id'].nunique():>2}경주               {med_d * 1000:6.1f}ms")
    print(f"   → 주말 실경기 최대 규모(3경마장 × 12경주 = 36경주) 추정 {med * 36 * 1000:.0f}ms")

    # 3. 슬라이더 — 유저가 비율을 바꿀 때. 여기만 동시 10명이 걸린다.
    rng = np.random.default_rng(0)
    race_ids = gm["race_id"].unique()[:12]                 # 방 하나에 띄우는 경주 수 정도
    sel = np.isin(gm["race_id"].to_numpy(), race_ids)
    sub_ax = gm_ax[sel]

    def one_slider():
        w = rng.dirichlet(np.ones(A))
        s = combine(sub_ax, w, 0.0, A)
        return s

    med_s, best_s = _timeit(one_slider, repeat=200)

    def ten_users():
        for _ in range(a.users):
            one_slider()

    med_u, _ = _timeit(ten_users, repeat=50)
    print(f"\n3) 슬라이더 재계산 (모델 안 돎 — 곱셈·덧셈만)")
    print(f"   1명 × {len(race_ids)}경주 {sel.sum():>3}두        {med_s * 1e6:6.0f}µs")
    print(f"   {a.users}명 동시                   {med_u * 1000:6.2f}ms")
    print(f"   게임풀 전체({len(gm):,}두) 1회    "
          f"{_timeit(lambda: combine(gm_ax, np.ones(A) / A, 0.0, A), repeat=20)[0] * 1000:6.2f}ms")

    print(f"\n[판정] 유저가 기다리는 경로는 3번뿐이고 {a.users}명이 동시에 눌러도 "
          f"{med_u * 1000:.2f}ms — 15초 기준의 {15000 / max(med_u * 1000, 1e-9):,.0f}분의 1.")
    print("       1·2번은 배포·주말 배치라 유저 요청 경로에 없다.")


if __name__ == "__main__":
    main()
