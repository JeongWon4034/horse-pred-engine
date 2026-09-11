"""출발 전 예측 — 오늘 아직 안 뛴 경주를 S3(seed 5) + LightGBM 73 으로 점수 내고 시각을 박아 남긴다.

  cd pipeline && PYTHONUTF8=1 uv run python -m ingest.ledger --date 20260911     # 오늘 출마표(미출발 행 포함) 갱신
  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.live prep 20260911      # 미출발 행에 임시 착순 → 빌더 통과
  cd pipeline && bash build_live.sh                                                # 팀 빌더로 피처 빌드 (new 분할)
  PYTHONUTF8=1 uv run --project pipeline python -m selfsup.live predict 20260911 3 # meet 3=부산경남

API214_1 은 경주 전에도 출마표 행(착순 빈칸·기록 0)을 준다. 그 행은 팀 빌더 load_ledger 의 `ord between 1..16`
필터에 걸려 사라지므로, prep 단계에서 **오늘 날짜의 착순 빈칸 행에만** 게이트 번호를 임시 착순으로 넣는다.
그 행의 y_* 는 쓰레기지만 예측 입력(as-of 피처)에는 안 들어가고, 같은 날 뒤 경주의 이력에도 안 들어간다(엄격히 이전 날짜만).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data as D

PIPE = D.PIPE
LEDGER = PIPE / "data" / "raw" / "ledger_api.csv"
LIVE = PIPE / "data" / "build" / "dataset" / "v2" / "model"
OUT = D.RUNS / "live"
OUT.mkdir(exist_ok=True)


def prep(day: str):
    d = pd.read_csv(LEDGER, dtype=str, low_memory=False)
    bak = LEDGER.with_name("ledger_api_orig.csv")
    if not bak.exists():
        d.to_csv(bak, index=False, encoding="utf-8-sig")
    m = (d["rcDate"].astype(str) == day) & (pd.to_numeric(d["ord"], errors="coerce").isna() | (d["ord"].astype(str).str.strip() == "") | (d["ord"].astype(str) == "0"))
    print(f"{day} 미출발 행 {int(m.sum())} (경주 {d.loc[m, 'rcNo'].nunique()}개) → 임시 착순 = 게이트")
    d.loc[m, "ord"] = d.loc[m, "chulNo"]
    d.loc[m, "_unrun"] = "1"
    d.to_csv(LEDGER, index=False, encoding="utf-8-sig")


def refresh(day: str):
    """발주 직전 갱신 — 오늘 행(날씨·함수율 T-10, 마체중 T-60~90, 끝난 경주 착순)을 API 에서 다시 받아
    원본 원장의 오늘 행을 통째로 바꾼 뒤 prep 을 다시 한다. 그 다음 build_live.sh → predict."""
    from ingest.ledger import key, fetch
    k = key(); rows = []
    for meet in (1, 2, 3):
        rows += fetch(k, f"meet={meet}&rc_date={day}")
    if not rows:
        raise SystemExit("오늘 행이 안 왔다")
    new = pd.DataFrame(rows).astype(str)
    new["meet"] = new["meet"].replace({"1": "서울", "2": "제주", "3": "부경"})
    bak = LEDGER.with_name("ledger_api_orig.csv")
    d = pd.read_csv(bak if bak.exists() else LEDGER, dtype=str, low_memory=False)
    keep = d[d["rcDate"].astype(str) != day]
    new = new.reindex(columns=d.columns, fill_value="")
    d2 = pd.concat([keep, new], ignore_index=True)
    d2.to_csv(bak, index=False, encoding="utf-8-sig")
    d2.to_csv(LEDGER, index=False, encoding="utf-8-sig")
    print(f"{day} 행 {len(d[d['rcDate'].astype(str) == day])} → {len(new)} 교체 ({time.strftime('%H:%M')})")
    prep(day)


def _pool_from_live(target_day: int) -> pd.DataFrame:
    """이력 풀을 실시간 빌드 parquet 에서 만든다 — 팀 사본은 5월까지라 여름 전적이 빠진다.
    target_day 이전 행만. game 도 넣는다(예측 서빙이지 평가가 아니다 — 정원 live 도 같은 판단)."""
    from model.history import POOL_COLS
    frames = []
    for sp in ("train", "valid", "test", "new", "game"):
        p = LIVE / f"{sp}.parquet"
        if p.exists():
            frames.append(D.S.clean(pd.read_parquet(p, columns=POOL_COLS)))
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[pool["rcDate"] < target_day]
    pool["hrNo"] = pool["hrNo"].astype(str); pool["jkNo"] = pool["jkNo"].astype(str)
    return pool.sort_values(["hrNo", "rcDate", "race_id"], kind="stable").reset_index(drop=True)


def _selfsup_probs(tr, tgt, cols, dev, epochs=14, seed=1):
    from .models import Body, RankHead
    from .run import seed_all, _pl
    from model.evaluate import win_probs
    import sys as _s
    from . import run as _run
    _s.modules["__main__"].pretrain = _run.pretrain
    ck = torch.load(D.RUNS / "mask_base.pt", weights_only=False)
    enc = ck["enc"]; va = D.load("valid")
    seed_all(seed)
    def T(df):
        r = D.pack_races(df, D.transform(df, enc))
        return r, {k: torch.from_numpy(v).to(dev) for k, v in
                   {"x": r.x, "mask": r.mask, "order": r.order}.items()}
    rtr, Ttr = T(tr); rva, Tva = T(va); rtg, Ttg = T(tgt)
    body = Body(enc.dim).to(dev); body.load_state_dict(ck["body"]); head = RankHead(64).to(dev)
    params = list(body.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-2)
    R = len(rtr); bs = 256
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=epochs * ((R + bs - 1) // bs), pct_start=0.25)
    pl = _pl(); best = (9e9, None)
    def score(Tx):
        body.eval(); head.eval()
        with torch.no_grad():
            return head(body(Tx["x"])).cpu().numpy()
    for ep in range(epochs):
        body.train(); head.train(); perm = torch.randperm(R, device=dev)
        for b in range(0, R, bs):
            i = perm[b:b + bs]
            loss = pl(head(body(Ttr["x"][i])), Ttr["mask"][i], Ttr["order"][i], topk=3)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
        ll = D.E.race_logloss(va, D.flatten_scores(score(Tva), rva))
        if ll < best[0]:
            best = (ll, D.flatten_scores(score(Ttg), rtg))
    print(f"selfsup 미세조정 valid logloss {best[0]:.4f}")
    return win_probs(tgt, best[1])


def predict(day: str, meet: str):
    from model.history import build as hist_build, fit_stats
    from model.categorical import EMBED_COLS, encode_cats, fit_vocabs
    from model.data import to_races
    from model.train import RUNS as CKPT_DIR, build as build_model, predict as dl_predict
    from model.evaluate import win_probs
    import lightgbm as lgb

    new = pd.read_parquet(LIVE / "new.parquet")
    new = D.S.clean(new)
    tgt = new[(new["rcDate"] == int(day)) & (new["race_id"].str.startswith(f"{day}_{meet}_"))].copy()
    tgt = tgt.sort_values(["race_id", "X_chulNo"], kind="stable").reset_index(drop=True)
    if tgt.empty:
        raise SystemExit(f"{day} meet={meet} 행이 new.parquet 에 없다")
    print(f"대상 {tgt['race_id'].nunique()}경주 {len(tgt)}두")

    tr = D.load("train"); cols = D.feature_cols()                 # 인코더·어휘는 학습 때(팀 사본)와 동일하게
    for c in cols:                                                # 육종가(F2_ebv_*) 는 실시간 재료가 없다 — 정원 live 와 같이 NaN
        if c not in tgt.columns:
            tgt[c] = np.nan
    tgt = tgt.copy()
    enc = D.fit_encoder(tr, cols)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ── S3 seed 5
    vocabs = fit_vocabs(tr, EMBED_COLS)
    pool = _pool_from_live(int(day))
    stats = fit_stats(pool, int(tr["rcDate"].max()))
    probs = []
    for seed in ("s20260901", "s1", "s2", "s3", "s4"):
        ck = torch.load(CKPT_DIR / f"history_73_L20_{seed}.pt", weights_only=False)
        assert enc.names == ck["names"]
        hist, n = hist_build(tgt, pool, stats, L=20)
        races = to_races(tgt, enc, {"cat": encode_cats(tgt, vocabs), "hist": hist, "hist_len": n})
        model = build_model(ck["kind"], enc, vocabs, dev); model.load_state_dict(ck["state"])
        T = {k: torch.as_tensor(v, device=dev) for k, v in races.extra.items()}
        from model.data import flatten_scores
        s = flatten_scores(dl_predict(model, races, T, dev), races)
        probs.append(win_probs(tgt, s))
    tgt["p_s3"] = np.mean(probs, axis=0)
    print(f"이력: 평균 {n.mean():.1f}/20, 0개 {np.mean(n == 0)*100:.0f}%")

    # ── LightGBM 73 + 마체중 3 (P7: logloss −0.009). 마체중은 당일 계체값 — refresh 로 최신화된다
    from .bodyweight import attach, FEATS as BW
    tr_bw, tg_bw = attach(tr), attach(tgt)
    print(f"마체중 충전: 오늘 {tg_bw['bw'].notna().mean()*100:.0f}%")
    tr_e, tg_e = D.C.encode(tr_bw, tg_bw, cols + BW)
    cols_lgb = cols + BW
    g = tr_e.groupby("race_id", sort=False).size().to_numpy()
    m = lgb.train(dict(objective="lambdarank", metric="ndcg", learning_rate=0.05, num_leaves=31,
                       min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                       verbose=-1, seed=0),
                  lgb.Dataset(tr_e[cols_lgb], label=tr_e["y_rel"], group=g), num_boost_round=200)
    tgt["p_lgb"] = win_probs(tgt, m.predict(tg_e[cols_lgb]))

    # ── selfsup: 마스크 사전학습 몸통(mask_base.pt) → 그 자리에서 미세조정(valid logloss 로 epoch 선택) → 점수
    tgt["p_ssl"] = _selfsup_probs(tr, tgt, cols, dev)

    tgt["p_ens"] = (tgt["p_s3"] + tgt["p_ssl"] + tgt["p_lgb"]) / 3          # 세 모델 확률 평균
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"# 출발 전 예측 — {day} meet {meet}", f"", f"예측 시각 **{stamp}** (KST). 모델: S3 L20 seed 5 평균 (05d8230 재측정본) · 사전학습(mask_base) 미세조정 seed 1 · LightGBM 73+마체중3 (재학습). 앙상블 = 세 확률 평균.",
             "이력 풀: 실시간 원장 빌드(train∪valid∪test∪new∪game, 당일 이전). 결과가 API 에 오르면 아래 표에 착순을 채운다.", ""]
    for rid, grp in tgt.groupby("race_id", sort=False):
        r = grp.sort_values("p_ens", ascending=False)
        top = r["p_ens"].to_numpy(); gap = (top[0] - top[1]) * 100 if len(top) > 1 else 0
        conf = "확신" if gap >= 10 else ("보통" if gap >= 4 else "혼전")
        lines += [f"## {rid}  ({int(grp['X_rcDist'].iloc[0])}m · {len(grp)}두 · 1-2위 격차 {gap:.1f}%p → {conf})", "",
                  "| 순위 | 마번 | 마명 | 앙상블 | S3 | 사전학습 | LGB | 실제 착순 |", "|---:|---:|---|---:|---:|---:|---:|---:|"]
        for k, (_, h) in enumerate(r.iterrows(), 1):
            lines.append(f"| {k} | {int(h['X_chulNo'])} | {h.get('hrName', '')} | {h['p_ens']*100:.1f}% | {h['p_s3']*100:.1f}% | {h['p_ssl']*100:.1f}% | {h['p_lgb']*100:.1f}% | |")
        lines.append("")
    hhmm = time.strftime("%H%M")
    out = OUT / f"pred_{day}_m{meet}_{hhmm}.md"                    # 시각별로 남긴다 — 덮어쓰지 않는다
    out.write_text("\n".join(lines), encoding="utf-8")
    cols_out = ["race_id", "hrNo", "X_chulNo", "p_s3", "p_ssl", "p_lgb", "p_ens"]
    tgt[cols_out].to_parquet(OUT / f"pred_{day}_m{meet}_{hhmm}.parquet", index=False)
    tgt[cols_out].to_parquet(OUT / f"pred_{day}_m{meet}.parquet", index=False)   # 최신본 (결과 대조용)
    print("\n".join(lines)); print(f"→ {out}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep(sys.argv[2])
    elif cmd == "refresh":
        refresh(sys.argv[2])
    elif cmd == "predict":
        predict(sys.argv[2], sys.argv[3])
