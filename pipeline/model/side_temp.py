"""사이드 실험 — 온도 조절(temperature scaling). 순위는 그대로 두고 확률 품질만 고친다.

    PYTHONUTF8=1 uv run python -m model.side_temp

점수를 T 로 나눈 뒤 경주 내 softmax 를 건다. 단조 변환이라 **top-1·top-3 은 바뀌지 않고**
logloss·ECE·Brier 만 움직인다. 재학습이 없다.

T 를 valid 전체로 고르면 그 숫자는 낙관값이다. 그래서 side_ensemble 과 같은 방식으로
경주를 반씩 갈라 교차 적합한다(한쪽에서 T 선택, 다른 쪽에서 측정).

**이 실험의 요점은 S3 가 좋아지는가가 아니다.** LightGBM 도 같은 보정을 받으면 S3 의
logloss 우위가 살아남는지가 판정 대상이다. 한쪽만 보정해서 이겼다고 쓰면 비교가 무효다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import compare, ece, hits, load_pred, race_logloss, win_probs
from .team import C

SEED = 20260901
L20_SEEDS = ["20260901", 1, 2, 3, 4]
GRID = np.round(np.arange(0.60, 2.01, 0.05), 2)


# ─────────────────────────── 온도를 씌운 확률 ───────────────────────────
def probs_at(df: pd.DataFrame, score_list: list[np.ndarray], T: float) -> np.ndarray:
    """점수를 T 로 나눠 경주 내 softmax. 여러 개면 확률 평균(seed 앙상블).

    seed 앙상블은 확률을 평균하므로 온도를 **평균 전 각 모델에** 씌운다. 평균 뒤에 씌우면
    softmax 를 두 번 거는 셈이 되어 정의가 달라진다.
    """
    return np.mean([win_probs(df, s / T) for s in score_list], axis=0)


def quality(df: pd.DataFrame, p: np.ndarray) -> dict:
    """확률만으로 재는 지표. log(p) 를 점수로 넣으면 win_probs 가 p 를 그대로 복원한다."""
    s = np.log(np.clip(p, 1e-12, 1))
    y = df["y_win"].to_numpy(float)
    return {"logloss": race_logloss(df, s), "ece": ece(df, s),
            "brier": float(((p - y) ** 2).mean())}


# ─────────────────────────── T 그리드 ───────────────────────────
def sweep(df: pd.DataFrame, name: str, score_list: list[np.ndarray]) -> float:
    """T 를 훑어 표를 찍고 logloss 최소 T 를 돌려준다 (valid 전체 = 낙관값).

    단일 모델은 T 로 나누는 것이 단조 변환이라 순위가 절대 안 바뀐다 — assert 로 못 박는다.
    **seed 앙상블은 다르다.** 확률을 평균하면 T 가 모델 간 상대 비중을 바꾸므로 argmax 가
    옮겨갈 수 있다. 그래서 앙상블에서는 top-1 도 같이 찍는다.
    """
    single = len(score_list) == 1
    base_hits = hits(df, np.log(probs_at(df, score_list, 1.0)))
    rows = []
    for T in GRID:
        p = probs_at(df, score_list, float(T))
        q = quality(df, p)
        h = hits(df, np.log(p))
        if single and not np.array_equal(h, base_hits):
            raise AssertionError(f"T={T} 에서 단일 모델 적중이 바뀌었다 — 온도 적용이 잘못됐다")
        rows.append((float(T), q["logloss"], q["ece"], q["brier"], h.mean() * 100))

    best = min(rows, key=lambda r: r[1])
    kind = "단조 — top-1 고정" if single else "확률 평균 — top-1 도 움직인다"
    print(f"\n[{name}]  T 그리드 (valid 전체 — 낙관값)   {kind}")
    print(f"{'T':>6}{'logloss':>10}{'ECE':>9}{'Brier':>9}{'top-1':>8}")
    for T, ll, e, b, t1 in rows:
        if T in (0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0) or T == best[0]:
            mark = "  ← 최소" if T == best[0] else ("  (현재)" if T == 1.0 else "")
            print(f"{T:>6.2f}{ll:>10.4f}{e:>9.4f}{b:>9.4f}{t1:>7.2f}%{mark}")
    return best[0]


# ─────────────────────────── 교차 적합 ───────────────────────────
def crossfit(df: pd.DataFrame, name: str, score_list: list[np.ndarray], reps: int = 20) -> dict:
    """경주를 반으로 갈라 한쪽에서 T 선택(logloss 최소), 다른 쪽에서 측정. reps 회 반복."""
    rid = df["race_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
    race_of_row = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(rid)]))

    # T 마다 확률을 한 번만 만들어 재사용한다 (반복마다 softmax 를 다시 걸지 않는다)
    P = {float(T): probs_at(df, score_list, float(T)) for T in GRID}

    rng = np.random.default_rng(SEED)
    picked, ll_out, ece_out, ll_base = [], [], [], []
    for _ in range(reps):
        perm = rng.permutation(len(starts))
        half = np.zeros(len(starts), bool)
        half[perm[: len(perm) // 2]] = True
        A = half[race_of_row]
        for fit, ev in ((A, ~A), (~A, A)):
            dfit, dev = df[fit], df[ev]
            T = min(GRID, key=lambda t: race_logloss(dfit, np.log(P[float(t)][fit])))
            q = quality(dev, P[float(T)][ev])
            picked.append(float(T))
            ll_out.append(q["logloss"]); ece_out.append(q["ece"])
            ll_base.append(race_logloss(dev, np.log(P[1.0][ev])))

    r = {"T_mean": float(np.mean(picked)), "T_min": min(picked), "T_max": max(picked),
         "logloss": float(np.mean(ll_out)), "logloss_base": float(np.mean(ll_base)),
         "ece": float(np.mean(ece_out))}
    print(f"\n[{name}]  교차 적합 ({reps}회 × 2분할)")
    print(f"  선택된 T 평균 {r['T_mean']:.2f} (범위 {r['T_min']:.2f}~{r['T_max']:.2f})")
    print(f"  홀드아웃 logloss {r['logloss']:.4f}  vs 보정 없음 {r['logloss_base']:.4f}"
          f"   → {r['logloss'] - r['logloss_base']:+.4f}")
    print(f"  홀드아웃 ECE     {r['ece']:.4f}")
    return r


def main() -> None:
    va = C.load("valid")
    lgb = load_pred(va, "lgb_73")
    if lgb is None:
        raise SystemExit("experiments/pred/lgb_73.parquet 이 없다 — model.baseline 을 먼저 돌린다")
    s3 = [load_pred(va, f"history_73_L20_s{s}") for s in L20_SEEDS]
    if any(s is None for s in s3):
        raise SystemExit("S3 L20 예측이 없다 — model.train history --hist-len 20 --tag _L20_s<seed>")

    print(f"valid {va['race_id'].nunique():,}경주 · T 그리드 {GRID[0]:.2f}~{GRID[-1]:.2f} "
          f"({len(GRID)}점)")
    print("단일 모델은 T 가 단조 변환이라 top-1 이 고정된다(assert). seed 앙상블은 확률 평균이라"
          " top-1 도 움직인다.")

    models = {
        "LightGBM 73": [lgb],
        "S3 L20 단일(seed 20260901)": [s3[0]],
        "S3 L20 seed 5개 평균": s3,
    }

    best_T, cf = {}, {}
    for name, sl in models.items():
        best_T[name] = sweep(va, name, sl)
    for name, sl in models.items():
        cf[name] = crossfit(va, name, sl)

    # ── 판정: 양쪽을 보정한 뒤에도 S3 의 logloss 우위가 남는가
    print("\n" + "=" * 78)
    print("[종합] 보정 전 → 보정 후 (교차 적합 홀드아웃 평균)")
    print(f"{'모델':<28}{'logloss 전':>11}{'logloss 후':>11}{'차이':>9}{'선택 T':>8}")
    for name in models:
        r = cf[name]
        print(f"{name:<28}{r['logloss_base']:>11.4f}{r['logloss']:>11.4f}"
              f"{r['logloss'] - r['logloss_base']:>+9.4f}{r['T_mean']:>8.2f}")

    lgb_after = cf["LightGBM 73"]["logloss"]
    for name in ("S3 L20 단일(seed 20260901)", "S3 L20 seed 5개 평균"):
        gap_before = cf[name]["logloss_base"] - cf["LightGBM 73"]["logloss_base"]
        gap_after = cf[name]["logloss"] - lgb_after
        print(f"\n{name} − LightGBM logloss")
        print(f"  보정 전 {gap_before:+.4f}   보정 후 {gap_after:+.4f}"
              f"   → 우위 {'유지' if gap_after < 0 else '소멸'}")

    # top-1 은 온도와 무관하므로 한 번만 적는다
    print("\n[참고] top-1 — 온도와 무관")
    h_lgb = hits(va, lgb)
    for name, sl in models.items():
        h = hits(va, np.log(probs_at(va, sl, 1.0)))
        t1 = h.mean() * 100
        if name == "LightGBM 73":
            print(f"  {name:<28}{t1:5.2f}%")
        else:
            d = compare(h, h_lgb)
            print(f"  {name:<28}{t1:5.2f}%   vs LGB {d[0]:+.2f}%p [{d[1]:+.2f}, {d[2]:+.2f}]")


if __name__ == "__main__":
    main()
