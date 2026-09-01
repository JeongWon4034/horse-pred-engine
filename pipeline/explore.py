# %% [markdown]
# # 데이터 탐색
#
# VS Code 에서 이 파일을 열면 각 `# %%` 블록 왼쪽에 **Run Cell** 이 뜬다.
# 셀 단위로 실행하면 결과가 오른쪽 인터랙티브 창에 뜨고, 변수가 메모리에 남는다.
#
# 주피터가 편하면: `uv run jupyter lab`

# %%
from eda import *

overview()

# %%
tr = load("train")
va = load("valid")
tr.shape, va.shape

# %% [markdown]
# ## 1. 컬럼 구성
# 97컬럼 = 식별자 10 + 타깃 5 + 피처 82

# %%
print("식별자:", IDS)
print("\n타깃  :", TARGETS)
print()
for g in GROUPS:
    cs = [c for c in tr.columns if c.startswith(g + "_")]
    print(f"{g:3} {GROUP_NAME[g]:<12} {len(cs):>2}개")
    print("     ", ", ".join(cs))

# %% [markdown]
# ## 2. 경주 하나 통째로 보기

# %%
race(tr, group="F1")          # 최근성적 축만
# race(tr)                    # 전부
# race(tr, "20251109_3_6")    # 특정 경주

# %% [markdown]
# ## 3. 결측
#
# 결측이 랜덤이 아니다. 세 가지 원인이 섞여 있다.
#  - **연도**: `F2_`(육종가) · `X_rating` 은 2016년 이전에 없음
#  - **신마**: 첫 출전이라 과거 기록이 없음 (6.9%)
#  - **첫 기승**: 그 기수가 그 말을 처음 탐 (39.7%)
#
# 값이 없는 게 아니라 **정보가 있는** 결측이라, 채우기 전에 플래그를 남겨야 한다.

# %%
missing(tr, by_group=True)

# %%
missing(tr, min_rate=0.01)

# %%
# 연도별 — 시점 때문에 비는 컬럼 찾기
plot_missing_by_year(tr, ["F2_ebv_prize", "X_rating", "F3_jkhr_win_rate", "F1_ord_avg3"])

# %%
# 신마 가설 검증
fs = tr.F1_first_start.fillna(0) == 1
print(f"신마 비율 {fs.mean()*100:.1f}%")
for c in ["F1_ord_avg3", "F5_style", "F1_layoff_days"]:
    print(f"  {c:<18} 신마 {tr.loc[fs, c].isna().mean()*100:5.1f}%  "
          f"비신마 {tr.loc[~fs, c].isna().mean()*100:5.1f}%")

# %% [markdown]
# ## 4. 범주형

# %%
cats(tr)

# %% [markdown]
# ## 5. 기초 통계

# %%
describe(tr, "F1")

# %% [markdown]
# ## 6. 분포 — 1착과 비1착이 갈리는지

# %%
plot_hist(tr, "F1_ordpct_avg5")

# %%
plot_hist(tr, "F6_mkt_prob")

# %% [markdown]
# ## 7. 피처 하나로 순위를 매기면 얼마나 맞나
#
# 그 피처가 혼자서 얼마나 센지 본다. 무작위는 약 9.5%.

# %%
signal(va).head(20)

# %% [markdown]
# ## 8. 게임용 데이터

# %%
rc = load_game("race_card")
en = load_game("entries")
po = load_game("payouts")
print(rc.shape, en.shape, po.shape)
po.head(10)

# %%
# 승식별 배당 분포
po.groupby("pool").odds.describe().round(1)

# %% [markdown]
# ## 9. 구간 통과순위 (★ 학습 금지 — 재생용)

# %%
ps = load_passing()
print(ps.point.unique())
ps[ps.race_id == ps.race_id.iloc[0]].pivot_table(
    index="chulNo", columns="point", values="pass_ord")
