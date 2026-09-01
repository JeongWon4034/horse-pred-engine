"""EDA 리포트를 그림으로 뽑는다.

    cd pipeline
    uv run python -m eda.report        # -> eda/report.html (브라우저로 열면 끝)

VS Code 인터랙티브 창이 안 뜨거나, 결과를 한 장으로 훑고 싶을 때 쓴다.
라이트/다크 두 벌을 렌더해서 넣으므로 어느 테마로 열어도 읽힌다.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eda import (  # noqa: E402
    GROUP_NAME, GROUPS, IDS, TARGETS,
    cats, features, load, load_game, load_passing, missing, missing_by_year, signal,
)

OUT = Path(__file__).resolve().parent / "report.html"

# 팔레트 — 라이트/다크 각각 그 바탕에서 검증된 단계를 쓴다
LIGHT = dict(bg="#fcfcfb", ink="#0b0b0b", sub="#52514e", grid="#dedcd6",
             s1="#2a78d6", s2="#eb6834", s3="#1baf7a", s4="#eda100", flat="#b9b7ae")
DARK = dict(bg="#1a1a19", ink="#ffffff", sub="#c3c2b7", grid="#3a3a37",
            s1="#3987e5", s2="#d95926", s3="#199e70", s4="#c98500", flat="#6a6963")

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False


# ────────────────────────────── 렌더 골격 ──────────────────────────────

def canvas(T, figsize=(9, 4.2), nrows=1, ncols=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(T["bg"])
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(T["bg"])
        ax.tick_params(colors=T["sub"], labelsize=9, length=0)
        for side, sp in ax.spines.items():
            sp.set_visible(side == "bottom")
            sp.set_color(T["grid"])
        ax.grid(color=T["grid"], alpha=.7, linewidth=.7)
        ax.set_axisbelow(True)
    return fig, axes


def finish(ax, T, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, color=T["ink"], fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel or "", color=T["sub"], fontsize=9)
    ax.set_ylabel(ylabel or "", color=T["sub"], fontsize=9)


def to_png(fig, T) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor=T["bg"])
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def figure(fn, note: str = "") -> str:
    """같은 그림을 라이트/다크로 한 벌씩 렌더해 <figure> 로 묶는다."""
    lm, dm = to_png(fn(LIGHT), LIGHT), to_png(fn(DARK), DARK)
    cap = f"<figcaption>{note}</figcaption>" if note else ""
    return (f'<figure class="chart">'
            f'<img class="lm" src="data:image/png;base64,{lm}" alt="">'
            f'<img class="dm" src="data:image/png;base64,{dm}" alt="">{cap}</figure>')


def table(df: pd.DataFrame, index=True) -> str:
    return f'<div class="tw">{df.to_html(index=index, border=0, justify="left")}</div>'


# ────────────────────────────── 그림들 ──────────────────────────────

def c_year_races(tr, va, te, gm):
    g = pd.DataFrame({k: d.groupby("yr").race_id.nunique()
                      for k, d in (("train", tr), ("valid", va), ("test", te),
                                   ("game", gm))}).fillna(0)

    def f(T):
        fig, ax = canvas(T, (9, 3.4))
        bottom = np.zeros(len(g))
        for k, c in (("train", T["s1"]), ("valid", T["s2"]), ("test", T["s3"]),
                     ("game", T["s4"])):
            ax.bar(g.index, g[k], bottom=bottom, color=c, label=k, width=.74,
                   edgecolor=T["bg"], linewidth=2)
            bottom += g[k].values
        ax.legend(frameon=False, labelcolor=T["sub"], fontsize=9, ncol=4,
                  loc="lower right", bbox_to_anchor=(1, 1.0))
        ax.set_xticks(g.index[::2], [str(y) for y in g.index[::2]])
        finish(ax, T, "연도별 경주 수 — 분할 넷이 어떻게 배치돼 있나", ylabel="경주")
        return fig
    return f


def c_group_missing(tr):
    m = missing(tr, by_group=True)

    def f(T):
        fig, ax = canvas(T, (9, 3.6))
        y = np.arange(len(m))
        ax.barh(y, m["결측률"] * 100, color=T["s1"], height=.62)
        ax.set_yticks(y, [f"{g}  {GROUP_NAME[g]}" for g in m.index], color=T["sub"])
        ax.invert_yaxis()
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        for i, (r, n) in enumerate(zip(m["결측률"], m["피처수"])):
            ax.text(r * 100 + .6, i, f"{r*100:.1f}%  ({n}개)", va="center",
                    color=T["sub"], fontsize=9)
        ax.set_xlim(0, max(m["결측률"]) * 100 * 1.35)
        finish(ax, T, "피처 축별 평균 결측률", xlabel="결측률 (%)")
        return fig
    return f


def c_missing_year(tr):
    cols = ["F2_ebv_prize", "X_rating", "F3_jkhr_win_rate", "F1_ord_avg3"]
    m = missing_by_year(tr, cols)

    def f(T):
        fig, ax = canvas(T, (9, 4.0))
        for c, col in zip(cols, [T["s1"], T["s2"], T["s3"], T["flat"]]):
            ax.plot(m.index, m[c] * 100, color=col, linewidth=2, marker="o", ms=4,
                    markeredgecolor=T["bg"], markeredgewidth=1.5, label=c)
        ax.legend(frameon=False, labelcolor=T["sub"], fontsize=9, ncol=2)
        ax.set_ylim(-3, 103)
        finish(ax, T, "연도별 결측률 — 시점 때문에 통째로 비는 컬럼", ylabel="결측률 (%)")
        return fig
    return f


def c_missing_cause(tr):
    fs = tr.F1_first_start.fillna(0) == 1
    fr = tr.F3_jkhr_starts.isna() | (tr.F3_jkhr_starts == 0)
    rows = [("신마", fs, ["F1_ord_avg3", "F5_style", "F1_layoff_days", "F1_speed_avg3"]),
            ("첫 기승", fr, ["F3_jkhr_win_rate"])]
    data = [(f"{c}  ({lab})",
             tr.loc[mask, c].isna().mean() * 100,
             tr.loc[~mask, c].isna().mean() * 100)
            for lab, mask, cs in rows for c in cs]
    rate_fs, rate_fr = fs.mean() * 100, fr.mean() * 100

    def f(T):
        fig, ax = canvas(T, (9, 3.8))
        y = np.arange(len(data))
        h = .36
        ax.barh(y - h / 2 - .01, [d[1] for d in data], height=h, color=T["s2"], label="해당")
        ax.barh(y + h / 2 + .01, [d[2] for d in data], height=h, color=T["flat"], label="비해당")
        ax.set_yticks(y, [d[0] for d in data], color=T["sub"])
        ax.invert_yaxis()
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        ax.set_xlim(0, 112)
        for i, d in enumerate(data):
            ax.text(d[1] + 1.5, i - h / 2 - .01, f"{d[1]:.0f}%", va="center",
                    color=T["sub"], fontsize=9)
            ax.text(d[2] + 1.5, i + h / 2 + .01, f"{d[2]:.0f}%", va="center",
                    color=T["sub"], fontsize=9)
        ax.legend(frameon=False, labelcolor=T["sub"], fontsize=9, loc="lower right")
        finish(ax, T, f"결측의 원인 — 신마 {rate_fs:.1f}% · 첫 기승 {rate_fr:.1f}%", xlabel="결측률 (%)")
        return fig
    return f


def c_top_missing(tr):
    s = missing(tr, min_rate=0.01)["결측률"].head(16)[::-1]

    def f(T):
        fig, ax = canvas(T, (9, 5.2))
        y = np.arange(len(s))
        ax.barh(y, s.values * 100, color=T["s1"], height=.66)
        ax.set_yticks(y, s.index, color=T["sub"])
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        for i, v in enumerate(s.values):
            ax.text(v * 100 + .6, i, f"{v*100:.1f}%", va="center", color=T["sub"], fontsize=9)
        ax.set_xlim(0, s.max() * 100 * 1.2)
        finish(ax, T, "결측률 상위 컬럼", xlabel="결측률 (%)")
        return fig
    return f


def c_hists(tr):
    cols = ["F1_ordpct_avg5", "F6_mkt_prob", "X_rating"]

    def f(T):
        fig, axes = canvas(T, (9.6, 2.9), 1, 3)
        for ax, c in zip(axes, cols):
            s = tr[c].dropna()
            lo, hi = s.quantile([.01, .99])
            for w, col, lab in ((0, T["flat"], "비1착"), (1, T["s2"], "1착")):
                ax.hist(tr.loc[tr.y_win == w, c].dropna(), bins=40, range=(lo, hi),
                        density=True, color=col, alpha=.72, label=lab)
            ax.set_yticks([])
            finish(ax, T, c)
        axes[0].legend(frameon=False, labelcolor=T["sub"], fontsize=9)
        fig.tight_layout()
        return fig
    return f


def c_signal(va):
    # _z / _rk 는 원본 피처를 경주 내에서 표준화·순위화한 것뿐이라 같은 값이 세 번 뜬다. 원본만 본다.
    cols = [c for c in features(va)
            if va[c].dtype.kind in "fi" and not c.endswith(("_z", "_rk"))]
    s = signal(va, cols).head(16)
    base = va.y_win.mean()

    def f(T):
        fig, ax = canvas(T, (9, 5.6))
        d = s[::-1]
        y = np.arange(len(d))
        ax.barh(y, d.top1 * 100, color=T["s1"], height=.66)
        ax.set_yticks(y, [f"{i}  ({r.방향})" for i, r in d.iterrows()], color=T["sub"])
        ax.axvline(base * 100, color=T["s2"], linewidth=2, linestyle="--")
        ax.set_ylim(-1.6, len(d) - .3)
        ax.text(base * 100 + .5, -1.15, f"무작위 {base*100:.1f}%",
                color=T["s2"], fontsize=9, va="center")
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        for i, (v, cv) in enumerate(zip(d.top1, d["커버"])):
            tag = "" if cv > .99 else f"  (커버 {cv*100:.0f}%)"
            ax.text(v * 100 + .5, i, f"{v*100:.1f}%{tag}", va="center",
                    color=T["sub"], fontsize=9)
        ax.set_xlim(0, d.top1.max() * 100 * 1.3)
        finish(ax, T, "피처 하나로만 1등을 골랐을 때 적중률 (valid)", xlabel="top-1 적중률 (%)")
        return fig
    return f


def c_field(tr):
    fs = tr.groupby("race_id", sort=False).size()
    vc = fs.value_counts().sort_index()

    def f(T):
        fig, ax = canvas(T, (9, 3.2))
        ax.bar(vc.index, vc.values, color=T["s1"], width=.72)
        ax.set_xticks(vc.index)
        finish(ax, T, f"경주당 출주 두수 — 중앙값 {int(fs.median())}두", xlabel="두수", ylabel="경주")
        return fig
    return f


def c_game_f2(g_f2, g_n):
    def f(T):
        fig, ax = canvas(T, (9, 3.4))
        col = [T["s2"] if v > .6 else T["s1"] for v in g_f2]
        ax.bar(g_f2.index, g_f2 * 100, color=col, width=.74)
        for y, v in g_f2.items():
            ax.text(y, v * 100 + 2.5, f"{g_n[y]}경주", ha="center", color=T["sub"], fontsize=8)
        ax.set_xticks(g_f2.index, [str(y) for y in g_f2.index])
        ax.set_ylim(0, 105)
        finish(ax, T, "게임풀의 육종가(F2) 결측률 — 주황은 60% 넘게 비는 해",
               ylabel="결측률 (%)")
        return fig
    return f


def c_odds(po):
    q = po.groupby("pool").odds.quantile([.25, .5, .75]).unstack()
    q = q.sort_values(.5)

    def f(T):
        fig, ax = canvas(T, (9, 3.4))
        y = np.arange(len(q))
        ax.hlines(y, q[.25], q[.75], color=T["flat"], linewidth=6)
        ax.plot(q[.5], y, "o", ms=9, color=T["s1"], markeredgecolor=T["bg"], markeredgewidth=2)
        ax.set_yticks(y, q.index, color=T["sub"])
        ax.set_xscale("log")
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        for i, v in enumerate(q[.5]):
            ax.text(v, i + .3, f"{v:,.0f}배", ha="center", color=T["sub"], fontsize=9)
        finish(ax, T, "승식별 배당 — 점은 중앙값, 막대는 25~75% 구간", xlabel="배당 (로그 축)")
        return fig
    return f


# ────────────────────────────── 조립 ──────────────────────────────

CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--sub:#52514e;--line:#e5e3dd;--accent:#2a78d6}
.dm{display:none}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){--bg:#131312;--card:#1a1a19;--ink:#ffffff;--sub:#c3c2b7;
    --line:#33332f;--accent:#3987e5}
  :root:not([data-theme="light"]) .lm{display:none}
  :root:not([data-theme="light"]) .dm{display:block}
}
:root[data-theme="dark"]{--bg:#131312;--card:#1a1a19;--ink:#ffffff;--sub:#c3c2b7;
  --line:#33332f;--accent:#3987e5}
:root[data-theme="dark"] .lm{display:none}
:root[data-theme="dark"] .dm{display:block}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.65 "Malgun Gothic","Segoe UI",system-ui,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:48px 24px 96px}
h1{font-size:28px;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:19px;margin:56px 0 6px;padding-top:24px;border-top:1px solid var(--line);
  letter-spacing:-.01em}
h3{font-size:14px;margin:28px 0 8px;color:var(--sub);font-weight:600}
p{color:var(--sub);margin:8px 0 18px;max-width:70ch}
.lede{font-size:16px}
b{color:var(--ink);font-weight:600}
code{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:1px 5px;
  font-size:13px;font-family:ui-monospace,Consolas,monospace}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:22px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.tile .k{font-size:12px;color:var(--sub);letter-spacing:.02em}
.tile .v{font-size:24px;font-weight:650;letter-spacing:-.02em;margin-top:2px;
  font-variant-numeric:tabular-nums}
.tile .m{font-size:12px;color:var(--sub);margin-top:2px;font-variant-numeric:tabular-nums}
.chart{margin:18px 0 8px}
.chart img{display:block;width:100%;height:auto;border:1px solid var(--line);
  border-radius:10px;background:var(--card)}
figcaption{font-size:13px;color:var(--sub);margin-top:8px}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card);
  margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:8px 12px;text-align:left;white-space:nowrap;border-bottom:1px solid var(--line)}
th{color:var(--sub);font-weight:600;font-size:12px}
tbody tr:last-child td{border-bottom:0}
td{font-variant-numeric:tabular-nums}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);
  border-radius:0 8px 8px 0;padding:12px 16px;margin:18px 0}
.note p{margin:0}
.foot{margin-top:64px;padding-top:16px;border-top:1px solid var(--line);font-size:12px;
  color:var(--sub)}
"""


def build() -> str:
    tr, va, te, gm = load("train"), load("valid"), load("test"), load("game")
    rc, en, po = load_game("race_card"), load_game("entries"), load_game("payouts")
    ps = load_passing()

    # 게임풀 격리 — 문서를 믿지 말고 매번 다시 센다
    def days(d):
        return set(zip(d.rcDate, d.meet))

    g_races, g_days = set(gm.race_id), days(gm)

    # 신마 결측이 실제로 얼마나 아픈지 — 숫자로 답한다
    fs = tr.F1_first_start.fillna(0) == 1
    win_fs, win_rest = tr.loc[fs, "y_win"].mean(), tr.loc[~fs, "y_win"].mean()
    n_fs = tr.groupby("race_id", sort=False).F1_first_start.apply(lambda s: (s.fillna(0) == 1).sum())
    size = tr.groupby("race_id", sort=False).size()
    race_has_fs, race_all_fs = (n_fs >= 1).mean(), (n_fs == size).mean()
    mid_lo, mid_hi = tr.loc[~fs, "F1_ordpct_avg5"].quantile([.4, .6])
    mid = tr[~fs & tr.F1_ordpct_avg5.between(mid_lo, mid_hi)]
    win_mid = mid.y_win.mean()

    # 게임풀에서 F2(육종가)가 얼마나 비는지 — 서비스에 그대로 나가는 값이다
    f2cols = [c for c in gm.columns if c.startswith("F2_")]
    g_f2 = gm.groupby("yr")[f2cols].apply(lambda d: d.isna().mean().mean())
    g_n = gm.groupby("yr").race_id.nunique()
    thin_yrs = g_f2[g_f2 > .6].index.tolist()
    thin_races = int(g_n[thin_yrs].sum()) if thin_yrs else 0
    iso = pd.DataFrame([{"분할": k,
                         "겹치는 경주": len(g_races & set(d.race_id)),
                         "겹치는 개최일": len(g_days & days(d))}
                        for k, d in (("train", tr), ("valid", va), ("test", te))]
                       ).set_index("분할")

    def ymd(v):
        s = str(int(v))
        return f"{s[:4]}.{s[4:6]}.{s[6:]}"

    tiles = "".join(
        f'<div class="tile"><div class="k">{n}</div><div class="v">{len(d):,}</div>'
        f'<div class="m">{d.race_id.nunique():,}경주<br>{ymd(d.rcDate.min())} ~ {ymd(d.rcDate.max())}'
        f"</div></div>"
        for n, d in (("train", tr), ("valid", va), ("test", te), ("game", gm)))
    tiles += (f'<div class="tile"><div class="k">피처</div><div class="v">{len(features(tr))}</div>'
              f'<div class="m">식별자 {len(IDS)} · 타깃 {len(TARGETS)}</div></div>')

    grp = pd.DataFrame([{"축": g, "이름": GROUP_NAME[g],
                         "피처 수": len([c for c in tr.columns if c.startswith(g + "_")])}
                        for g in GROUPS]).set_index("축")

    parts = [
        f"<title>경주 데이터 EDA</title>",
        f"<style>{CSS}</style>",
        '<div class="wrap">',
        "<h1>경주 데이터 EDA</h1>",
        f'<p class="lede">학습에 들어가는 {len(features(tr))}개 피처가 어떤 모양인지, '
        "어디가 비어 있고 왜 비는지, 혼자서 힘을 쓰는 피처가 뭔지를 한 장에 모았다.</p>",
        f'<div class="tiles">{tiles}</div>',

        "<h2>1. 분할 넷과 게임풀 격리</h2>",
        "<p>train 뒤를 시간순으로 잘라 valid·test 를 만들었다. 최근 구간이 얇다는 뜻이라, "
        "검증 결과를 볼 때 표본 수를 같이 봐야 한다. 여기에 <b>서비스에 내보낼 게임풀</b>이 "
        "네 번째 분할로 따로 있다 — 2025~2026년은 주말 실시간 예측용으로 학습에 넣어야 해서, "
        "게임풀은 뒤쪽 시간 블록이 아니라 2015~2024에 흩어서 예약했다.</p>",
        figure(c_year_races(tr, va, te, gm),
               f"게임풀 {len(g_races):,}경주가 {len(g_days)}개 개최일에 흩어져 있다."),
        '<div class="note"><p>게임풀은 <b>개최일 단위로 통째로</b> 뺐다. 경주 단위로 흩뿌리면 '
        "같은 날 앞뒤 경주가 학습에 남아 주로 상태·날씨가 새어 나간다. 아래는 문서를 믿지 않고 "
        "리포트를 만들 때마다 다시 센 결과다.</p></div>",
        table(iso),
        ("<p>경주 단위도 개최일 단위도 <b>전부 0건</b>. 모델이 게임 경주를 본 적이 없으니 "
         "‘내 모델 vs 기본 모델 vs 시장’ 비교가 성립한다.</p>"
         if int(iso.to_numpy().sum()) == 0 else
         f'<div class="note"><p><b>격리가 깨졌다.</b> 겹침 {int(iso.to_numpy().sum()):,}건. '
         "게임풀이 학습에 섞였으니 데이터셋을 다시 빌드하기 전에는 평가 숫자를 믿으면 안 된다."
         "</p></div>"),

        "<h2>2. 컬럼 구성</h2>",
        "<p>7개 축으로 나뉜다. 축 이름이 곧 컬럼 접두사다.</p>",
        table(grp),

        "<h2>3. 결측 — 세 가지 원인이 섞여 있다</h2>",
        '<div class="note"><p>결측이 랜덤이 아니다. <b>시점</b>(2016년 이전엔 육종가·레이팅 지표가 '
        "존재하지 않음), <b>신마</b>(첫 출전이라 과거 기록이 없음), <b>첫 기승</b>(그 기수가 그 말을 "
        "처음 탐). 값이 없는 게 아니라 <b>정보가 있는</b> 결측이라, 채우기 전에 플래그를 남겨야 한다."
        "</p></div>",
        figure(c_group_missing(tr), "축별 평균 결측률. 어느 축을 손봐야 하는지가 여기서 갈린다."),
        "<h3>원인 ① 시점</h3>",
        figure(c_missing_year(tr),
               "특정 연도 이전이 통째로 100%. 수집이 빠진 게 아니라 그때는 존재하지 않던 지표다."),
        "<h3>원인 ② · ③ 신마와 첫 기승</h3>",
        figure(c_missing_cause(tr), "해당 그룹에서만 결측이 100%로 튄다. 결측이 랜덤이 아니라는 증거."),
        f"<p>다만 <b>신마 결측은 겁낼 것이 못 된다</b>. 신마 1착률 {win_fs*100:.1f}% vs "
        f"비신마 {win_rest*100:.1f}% 로 차이가 {abs(win_fs-win_rest)*100:.1f}%p뿐이고, "
        f"신마가 한 두라도 낀 경주는 {race_has_fs*100:.0f}%, 전원 신마인 경주는 "
        f"{race_all_fs*100:.1f}%에 그친다. LightGBM은 NaN을 그대로 학습하고 "
        "<code>F1_first_start</code> 플래그도 이미 컬럼에 있으니, <b>채우지 말고 두는 것</b>이 정답에 "
        f"가깝다. 굳이 평균으로 채워도 큰 사고는 안 난다 — 실제 평균권(40~60%) 말의 1착률이 "
        f"{win_mid*100:.1f}%라 신마와 사실상 같은 자리에 놓이기 때문이다. "
        "<b>진짜 문제는 F2(육종가)</b>다. 아래를 보라.</p>",
        "<h3>결측률 상위 컬럼</h3>",
        figure(c_top_missing(tr)),

        "<h2>4. 분포 — 1착과 비1착이 갈리는가</h2>",
        "<p>두 분포가 겹칠수록 그 피처 하나로는 못 가른다는 뜻이다.</p>",
        figure(c_hists(tr)),

        "<h2>5. 피처 하나로 순위를 매기면 얼마나 맞나</h2>",
        f"<p>피처 하나로 경주 내 1등을 찍었을 때의 적중률. 무작위는 {va.y_win.mean()*100:.1f}%. "
        "커버는 그 피처로 판단이 가능했던 경주 비율이다 — 커버가 낮으면 적중률이 높아도 못 믿는다.</p>",
        figure(c_signal(va)),

        "<h2>6. 경주의 모양</h2>",
        figure(c_field(tr)),

        "<h2>7. 게임용 데이터</h2>",
        f"<p>출마표 {len(rc):,}행 · 출전 {len(en):,}행 · 배당 {len(po):,}행.</p>",
        figure(c_game_f2(g_f2, g_n)),
        ('<div class="note"><p><b>서비스에서 걸릴 지점.</b> 게임풀은 2015년부터라 '
         f'{", ".join(map(str, thin_yrs))}년 경주({thin_races:,}경주, 전체의 '
         f"{thin_races/len(g_races)*100:.0f}%)는 육종가 축이 사실상 통째로 빈다. "
         "유저가 그 경주를 뽑으면 다른 해와 같은 조건에서 예측하는 게 아니다. "
         "경주를 무작위로 내보낼 거면 <b>연도 가중치를 주거나 2017년 이후로 제한</b>하는 편이 낫다."
         "</p></div>") if thin_yrs else "",
        figure(c_odds(po), "승식이 어려워질수록 꼬리가 길어진다. 로그 축이라는 점에 주의."),

        "<h2>8. 범주형</h2>",
        table(cats(tr)),

        "<h2>9. 구간 통과순위 (★ 학습 금지 — 재생용)</h2>",
        f'<p>지점 {", ".join(map(str, ps.point.unique()))} 에서의 통과순위. 경주가 끝나야 알 수 있는 '
        "값이라 피처로 쓰면 누수다. 경주 재생 화면에만 쓴다. 아래는 한 경주 샘플.</p>",
        table(ps[ps.race_id == ps.race_id.iloc[0]]
              .pivot_table(index="chulNo", columns="point", values="pass_ord")),

        '<div class="foot">재생성: <code>uv run python -m eda.report</code> · '
        f"train {len(tr):,}행 기준</div>",
        "</div>",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(f"완료 -> {OUT}  ({OUT.stat().st_size / 1024 / 1024:.1f} MB)")
