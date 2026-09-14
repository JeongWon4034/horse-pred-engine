# -*- coding: utf-8 -*-
"""배포 산출물 생성 — 축 점수 테이블 · 프리셋 · 온도 T.

    uv run python -m basemodel.export --ckpt artifacts/runs/axis_77_s20260901.pt

기능명세서 AI-05(게임풀 점수 사전 계산) · AI-06(브라우저 가중합) 을 만족시키는 자리다.
**운영 중에는 모델이 돌지 않는다.** 여기서 3,700경주 점수를 한 번 계산해 DB 에 넣고,
유저가 슬라이더를 만지면 화면이 곱셈·덧셈·softmax 만 한다.

산출물
  axis_scores.parquet   말당 축 점수 6(+시장) — BE 가 DB 에 적재
  meta.json             프리셋 5종 가중치 · 온도 T · 축 메타 · 검증 수치 — FE 상수
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from . import presets as P
from .evaluate import ece, metrics, race_logloss, race_znorm, upset_metrics, win_probs
from .serve import Scorer, combine, load_game
from .team import ARTIFACTS, C


# ─────────────────────────── 온도 T ───────────────────────────
def fit_temperature(df: pd.DataFrame, raw: np.ndarray,
                    grid=np.arange(0.2, 6.01, 0.05)) -> float:
    """softmax(raw × T) 의 1착 확률이 실제와 맞도록 T 를 고른다 (temperature scaling).

    왜 따로 저장하나 — 축 점수는 경주 내 z-score 라 '순위'는 맞지만 '확률의 뾰족함'은
    담고 있지 않다. 유저 가중치 합이 100 이라는 화면 계약 때문에 스케일이 한 번 더
    깎이므로, 정규화된 점수를 확률로 되돌리는 스칼라 하나를 **검증셋에서** 맞춰 둔다.
    학습셋에서 맞추면 in-sample 점수라 과신한다.
    """
    best = (np.inf, 1.0)
    for t in grid:
        ll = race_logloss(df, raw * t)
        if ll < best[0]:
            best = (ll, float(t))
    return best[1]


def calibration_table(df: pd.DataFrame, axis_rows: np.ndarray, n_axes: int,
                      T: float, probes: dict[str, np.ndarray], znorm: bool) -> list[dict]:
    """가중치를 극단으로 밀어도 T 하나로 확률이 버티는지 — 계약의 약점을 드러내는 표."""
    out = []
    y = df["y_win"].to_numpy(float)
    for name, w in probes.items():
        raw = combine(axis_rows, w, 0.0, n_axes)
        if znorm:
            raw = race_znorm(df, raw)
        p = win_probs(df, raw * T)
        hi = p >= np.quantile(p, 0.95)
        out.append({"weights": name, "mean_p": float(p.mean() * 100),
                    "logloss": race_logloss(df, raw * T), "ece": ece(df, raw * T),
                    "top_bin_pred": float(p[hi].mean() * 100),
                    "top_bin_act": float(y[hi].mean() * 100)})
    return out


# ─────────────────────────── 점수 테이블 ───────────────────────────
def score_table(df: pd.DataFrame, axis_rows: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """BE 가 DB 에 넣을 표. 축 점수는 경주 내 z-score 원값을 그대로 싣는다.

    0~100 백분위로 바꾸지 않는 이유 — 백분위는 '얼마나 더 좋은지'를 지운다. 3σ 앞선 말과
    1σ 앞선 말이 같은 값이 되어 확률이 무너진다. 화면 표시용 0~100 은 `disp_*` 로 따로 낸다.
    """
    out = pd.DataFrame({
        "race_id": df["race_id"].to_numpy(),
        "hrNo": df["hrNo"].to_numpy(),
        "chulNo": df["X_chulNo"].to_numpy(),
    })
    for i, c in enumerate(columns):
        out[f"ax_{c.lower()}"] = axis_rows[:, i].astype(np.float32)
    # 표시용 — 경주 내 백분위 0~100. 계산에는 쓰지 않는다.
    for i, c in enumerate(columns):
        s = pd.Series(axis_rows[:, i], index=out.index)
        out[f"disp_{c.lower()}"] = (s.groupby(out["race_id"]).rank(pct=True) * 100).astype(np.float32)
    for extra in ("y_ord", "y_win", "F6_mkt_prob"):
        if extra in df.columns:
            out[extra] = df[extra].to_numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=str(ARTIFACTS / "export"))
    ap.add_argument("--samples", type=int, default=4000, help="프리셋 가중치 후보 수")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-game", action="store_true")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = Scorer(Path(a.ckpt), device=a.device)
    A, cols = sc.n_axes, sc.columns
    print(f"[체크포인트] {a.ckpt}\n  축 {cols}  시장축 {'있음' if sc.market else '없음'}  device={a.device}")

    # ── 축 점수 ─────────────────────────────────────────────────────
    tr_df, va_df = C.load("train"), C.load("valid")
    t0 = time.time()
    tr_ax = sc.axis_rows(tr_df, "train")
    va_ax = sc.axis_rows(va_df, "valid")
    print(f"[점수] train {tr_ax.shape} / valid {va_ax.shape}  ({time.time() - t0:.1f}s)")

    # ── 프리셋: train 에서 맞추고 valid 에서 보고 ──────────────────────
    obj_tr = P.Objective(tr_df, tr_ax, A)
    obj_va = P.Objective(va_df, va_ax, A)
    presets = P.fit_all(obj_tr, sc.axes, n_samples=a.samples, allow_market=sc.market,
                        obj_eval=obj_va)
    print("\n[프리셋 — train 에서 적합]")
    print(P.report(presets, tr_df, tr_ax, A, sc.axes))
    print("\n[프리셋 — valid 에서 측정 (보고용 정직한 값)]")
    print(P.report(presets, va_df, va_ax, A, sc.axes))
    for name, pr in presets.items():
        if "curve" in pr:
            c = pr["curve"]
            where = "valid" if pr.get("fit_on") == "eval" else "train"
            print(f"\n[{pr['label']} — 시장 계수 m 곡선 ({where}, 목적 = 고배당 적중률 %)]")
            step = max(len(c) // 16, 1)
            print("  " + "".join(f"{x['m']:>7.2f}" for x in c[::step]))
            print("  " + "".join(f"{x['raw']:>7.2f}" for x in c[::step]) + "   원값")
            print("  " + "".join(f"{x['smooth']:>7.2f}" for x in c[::step]) + "   평활")
            print(f"  → 채택 m = {pr['m']:.3f}")

    print("\n[교차적합 확인 — valid 를 반 갈라 한쪽에서 맞추고 다른 쪽에서 측정]")
    print(P.crossfit_report(va_df, va_ax, A, sc.axes, n_samples=max(a.samples // 2, 500),
                            allow_market=sc.market))

    # ── 온도 T ──────────────────────────────────────────────────────
    eq = np.full(A, 1 / A)
    probes = {"균등": eq}
    for i, ax in enumerate(sc.axes):
        w = np.zeros(A); w[i] = 1.0
        probes[f"{cfg.AXIS_KO[ax]} 100"] = w

    # 두 계약을 같은 자로 재고 고른다.
    #   raw    = 가중합 그대로 softmax        — 계산이 한 단계 적다
    #   znorm  = 경주 내 표준화 후 softmax    — T 가 가중치와 무관해진다
    chosen = {}
    for variant, zn in (("raw", False), ("znorm", True)):
        base_raw = combine(va_ax, eq, 0.0, A)
        T_v = fit_temperature(va_df, race_znorm(va_df, base_raw) if zn else base_raw)
        rows = calibration_table(va_df, va_ax, A, T_v, probes, zn)
        worst = max(abs(r["top_bin_pred"] - r["top_bin_act"]) for r in rows)
        chosen[variant] = {"T": T_v, "rows": rows, "worst_gap": worst,
                           "mean_logloss": float(np.mean([r["logloss"] for r in rows]))}
        print(f"\n[온도 T — {variant}] {T_v:.3f}   극단 가중치 최악 오차 {worst:.1f}%p   "
              f"평균 logloss {chosen[variant]['mean_logloss']:.4f}")
        print(f"  {'가중치':<12}{'평균확률':>9}{'logloss':>10}{'ECE':>8}{'상위5% 예측':>12}{'상위5% 실제':>12}")
        for r in rows:
            print(f"  {r['weights']:<12}{r['mean_p']:>8.1f}%{r['logloss']:>10.4f}{r['ece']:>8.4f}"
                  f"{r['top_bin_pred']:>11.1f}%{r['top_bin_act']:>11.1f}%")

    variant = min(chosen, key=lambda k: chosen[k]["worst_gap"])
    T = chosen[variant]["T"]
    print(f"\n  → 채택: {variant}  (극단 가중치에서 확률이 덜 무너지는 쪽)")

    # ── 게임풀 사전 계산 ─────────────────────────────────────────────
    game_rows = 0
    if not a.skip_game:
        gm_df = load_game()
        t0 = time.time()
        gm_ax = sc.axis_rows(gm_df, "game")
        dt = time.time() - t0
        tbl = score_table(gm_df, gm_ax, cols)
        tbl.to_parquet(out_dir / "axis_scores.parquet", index=False)
        game_rows = len(tbl)
        n_races = gm_df["race_id"].nunique()
        print(f"\n[게임풀] {n_races:,}경주 {len(tbl):,}두 점수 계산 {dt:.1f}s "
              f"({dt / n_races * 1000:.2f}ms/경주)  → axis_scores.parquet "
              f"({(out_dir / 'axis_scores.parquet').stat().st_size / 1e6:.1f}MB)")
        for name, p in presets.items():
            s = combine(gm_ax, p["w"], p["m"], A)
            t = C.evaluate(gm_df, s)
            u = upset_metrics(gm_df, s)
            print(f"    {p['label']:<8} top1 {t['top1']:5.1f}%  top3 {t['top3']:5.1f}%  "
                  f"역배적중 {u['upset_hit']:4.1f}%  평균배당 {u['avg_odds']:5.1f}")
        print("    ⚠ 게임풀은 학습 시기(2015~2024)와 섞여 있어 valid 보다 낙관적이다. 표기는 valid 값으로.")

    # ── meta.json ───────────────────────────────────────────────────
    eq_metrics = metrics(va_df, combine(va_ax, eq, 0.0, A), with_upset=True)
    meta = {
        "model": "AxisRanker",
        "checkpoint": Path(a.ckpt).name,
        "n_features": 77 if sc.market else 73,
        "axes": [{"key": ax, "label": cfg.AXIS_KO[ax], "order": i}
                 for i, ax in enumerate(sc.axes)],
        "market_axis": cfg.MARKET if sc.market else None,
        "temperature": T,
        "normalize_within_race": variant == "znorm",
        "score_contract": (
            "raw = (Σ_k groupWeights[k] * ax_k) / 100 + market * ax_market"
            + (" ; z = (raw - mean_race(raw)) / (std_race(raw) + 1e-6) ; p = softmax(z * temperature) 경주 내"
               if variant == "znorm" else
               " ; p = softmax(raw * temperature) 경주 내")),
        "calibration": {k: {"T": v["T"], "worst_gap_pp": round(v["worst_gap"], 2),
                            "mean_logloss": round(v["mean_logloss"], 4)}
                        for k, v in chosen.items()},
        "presets": {
            name: {"label": p["label"], "desc": p["desc"], "objective": p["objective"],
                   "groupWeights": p["groupWeights"], "market": round(p["m"], 4),
                   "fitOn": p.get("fit_on", "train"),
                   "replayOnly": name in cfg.REPLAY_ONLY_PRESETS}
            for name, p in presets.items()},
        "validation": {
            "split": "valid", "n_races": int(va_df["race_id"].nunique()),
            "equal_weight": {k: round(v, 4) for k, v in eq_metrics.items()},
            "presets": {name: {**{k: round(v, 3) for k, v in
                                  C.evaluate(va_df, combine(va_ax, p["w"], p["m"], A)).items()},
                               **{k: round(v, 3) for k, v in
                                  upset_metrics(va_df, combine(va_ax, p["w"], p["m"], A)).items()}}
                        for name, p in presets.items()}},
        "upset_odds_floor": cfg.UPSET_ODDS,
        "game_rows": game_rows,
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {out_dir}/meta.json")


if __name__ == "__main__":
    main()
