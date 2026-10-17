# 決定論的・ノンパラメトリック・フロンティア分析サーベイ

**作成日:** 2026-05-07
**位置付け:** [`ml_sfa_survey.md`](./ml_sfa_survey.md) が確率的フロンティア (SFA) を扱うのに対し、
本ドキュメントは決定論的・ノンパラメトリック系の効率性分析手法（DEA, FDH, COLS, StoNED, EAT 等）を網羅する。
教科書としても利用可能な水準で定式化・推定・性質・実装含意を整理し、
将来の `src/ml_sfa/frontier/` 配下の実装と双方向リンクすることを意図する。
**関連:** 用語定義は [`../glossary.md`](../glossary.md)、設計意思決定の記録は [`../../HISTORY.md`](../../HISTORY.md) (P-0001)。

---

## 目次

1. [フロンティア分析の全体像](#1-フロンティア分析の全体像)
2. [基礎理論: 生産可能集合と距離関数](#2-基礎理論-生産可能集合と距離関数)
3. [DEA: Data Envelopment Analysis](#3-dea-data-envelopment-analysis)
4. [FDH: Free Disposal Hull とロバストフロンティア](#4-fdh-free-disposal-hull-とロバストフロンティア)
5. [半パラメトリック・決定論的手法 (COLS / MOLS / TFA / DFA)](#5-半パラメトリック決定論的手法-cols--mols--tfa--dfa)
6. [StoNED と CNLS: DEA と SFA の橋渡し](#6-stoned-と-cnls-dea-と-sfa-の橋渡し)
7. [ML 系ノンパラメトリック手法 (EAT / RF / Boosting)](#7-ml-系ノンパラメトリック手法-eat--rf--boosting)
8. [動学・生産性指数](#8-動学生産性指数)
9. [望ましくない産出と方向距離関数](#9-望ましくない産出と方向距離関数)
10. [手法間の体系的比較](#10-手法間の体系的比較)
11. [動作検証用データセット](#11-動作検証用データセット)
12. [本ライブラリにおける実装計画 (将来)](#12-本ライブラリにおける実装計画-将来)
13. [参考文献](#13-参考文献)

---

## 1. フロンティア分析の全体像

### 1.1 効率性測定の二大潮流

効率性分析の手法は **誤差項に確率成分を含めるか否か** で大別される。

```
効率性分析の系譜
├── 決定論的フロンティア (Deterministic Frontier)
│   ├── ノンパラメトリック
│   │   ├── DEA (CCR, BCC, SBM, ...)        ← §3
│   │   ├── FDH (凸性を緩和)                ← §4
│   │   └── Order-m / Order-α (ロバスト化) ← §4.3
│   └── パラメトリック
│       ├── COLS / MOLS (OLS 補正)         ← §5.1
│       └── 線形計画 (Aigner-Chu 1968)     ← §5.0
│
├── 確率的フロンティア (Stochastic Frontier, SFA)        ← `ml_sfa_survey.md`
│   ├── パラメトリック SFA (ALS 1977 系)
│   └── ML-SFA (NN, GP, BART 等)
│
└── ハイブリッド (両者の橋渡し)
    ├── StoNED (CNLS + 誤差分解)            ← §6
    ├── Thick Frontier Approach (TFA)       ← §5.3
    └── Distribution-Free Approach (DFA)    ← §5.4
```

### 1.2 軸による分類マトリクス

| 軸 / 手法 | DEA | FDH | COLS | StoNED | SFA | EAT |
|-----------|:---:|:---:|:----:|:------:|:---:|:---:|
| 関数形仮定 | なし | なし | あり | shape のみ | あり | なし |
| 確率誤差 | × | × | △ (一部) | ○ | ○ | × |
| 凸性仮定 | ○ | × | ○ | ○ | (関数依存) | × |
| 単調性 | ○ | ○ | ○ | ○ | (関数依存) | ○ |
| 計算量 | LP | 比較 | OLS | QP+MLE | MLE | 木学習 |
| 不確実性定量化 | ブートストラップ | 同左 | 部分的 | ○ | ○ | △ |

### 1.3 本サーベイの守備範囲

確率成分を**含まない**または**間接的にしか含まない**手法群が本ドキュメントの中心である。
SFA 系（ALS 1977 以降の合成誤差モデル、ML-SFA 拡張）は [`ml_sfa_survey.md`](./ml_sfa_survey.md) に委ねる。
StoNED は CNLS という凸ノンパラメトリック回帰の上に SFA 的な誤差分解を載せるため、
本ドキュメントと SFA サーベイの**橋渡し**として §6 で扱う。

---

## 2. 基礎理論: 生産可能集合と距離関数

### 2.1 生産可能集合 (Production Possibility Set)

`N` 個の意思決定単位 (Decision Making Unit, DMU) があり、各 DMU `i` が
入力 `x_i ∈ R_+^p` を用いて産出 `y_i ∈ R_+^q` を得るとする。
**生産可能集合** `T ⊂ R_+^{p+q}` を以下で定義する:

```
T = { (x, y) | x で y が生産可能 }
```

経済学では `T` に対して以下の公理が課される:

| 公理 | 内容 |
|------|------|
| A1: フリーディスポーザリティ | `(x, y) ∈ T`, `x' ≥ x`, `y' ≤ y` ⇒ `(x', y') ∈ T` |
| A2: 凸性 | `T` は凸集合 (DEA で仮定、FDH では仮定しない) |
| A3: 規模に関する仮定 | CRS (Constant Returns to Scale) / VRS / NIRS / NDRS |
| A4: 閉性 | `T` は閉集合 |

### 2.2 効率的境界 (Efficient Frontier)

DMU `(x_0, y_0) ∈ T` が**入力指向で効率的**とは、
`y_0` を維持しつつ `x_0` の任意成分を厳密に減らすと `T` の外に出ること。
形式的には:

```
EFF_in(T) = { (x, y) ∈ T | ∀ x' ≤ x, x' ≠ x ⇒ (x', y) ∉ T }
```

産出指向の効率境界 `EFF_out(T)` は同様に `y` の拡張で定義する。

### 2.3 Farrell 効率性 (1957)

**Farrell (1957)** は効率性を以下の三層に分解した。

| 効率性 | 定義 | 識別 |
|--------|------|------|
| Technical Efficiency (TE) | フロンティア上に到達できているか | 投入物比率を維持して縮小可能か |
| Allocative Efficiency (AE) | 投入比率が価格に照らして最適か | 価格情報が必要 |
| Cost / Revenue Efficiency | TE × AE | 双方の積 |

入力指向 Farrell 技術効率性 `θ_F`:

```
θ_F(x_0, y_0) = min { θ > 0 | (θ x_0, y_0) ∈ T }
```

`θ_F ∈ (0, 1]`。`θ_F = 1` は完全効率を意味する。

### 2.4 Shephard 距離関数

Shephard (1953) は双対的な距離関数を導入した。

**入力距離関数:**

```
D_I(x, y) = sup { λ > 0 | (x / λ, y) ∈ T } = 1 / θ_F^in(x, y)
```

**産出距離関数:**

```
D_O(x, y) = inf { φ > 0 | (x, y / φ) ∈ T }
```

産出指向 Farrell 効率性は `φ_F = 1 / D_O ≥ 1` で、`φ_F = 1` が完全効率。

### 2.5 方向距離関数 (Directional Distance Function)

**Chambers, Chung, Färe (1996, 1998)** は望ましくない産出を扱うために方向距離関数を導入した。
方向ベクトル `g = (-g_x, g_y) ∈ R^p × R^q` (`g_x ≥ 0`, `g_y ≥ 0`) を与え:

```
D_g(x, y) = sup { β | (x - β g_x, y + β g_y) ∈ T }
```

DDF は加法的な効率性表現を与え、後述の SBM や非望産出モデルの基礎となる (§9)。

### 2.6 規模の仮定 (CRS / VRS / NIRS / NDRS)

凸結合に課す制約により T は変化する。

| 仮定 | 制約 (DEA の双対乗数) | 経済学的解釈 |
|------|---------------------|---------------|
| CRS | `Σ λ_j` 制約なし | 全 DMU が最適規模で操業 |
| VRS | `Σ λ_j = 1` | 規模に関する仮定なし |
| NIRS | `Σ λ_j ≤ 1` | 非増加収穫 |
| NDRS | `Σ λ_j ≥ 1` | 非減少収穫 |

CRS と VRS の効率性比 `θ_CCR / θ_BCC` は**規模効率性 (Scale Efficiency, SE)** と呼ばれる。

> **実装リンク (将来):** §12 を参照。`src/ml_sfa/frontier/_technology.py` 想定。

---

## 3. DEA: Data Envelopment Analysis

### 3.1 歴史的経緯

DEA は **Charnes, Cooper, Rhodes (1978, EJOR)** により Farrell (1957) の二次元測定を多入力多産出に拡張した手法として誕生した。
**Banker, Charnes, Cooper (1984, MS)** が VRS への拡張 (BCC モデル) を行い、以後 40 年以上にわたり多数の派生モデルが提案されている。

### 3.2 CCR モデル (CRS, 比率形式)

DMU `0` の効率性 `h_0` を以下の分数計画問題で定義する。

```
max  h_0 = (Σ_r u_r y_{r0}) / (Σ_i v_i x_{i0})
s.t. (Σ_r u_r y_{rj}) / (Σ_i v_i x_{ij}) ≤ 1   ∀ j = 1, ..., n
     u_r ≥ 0, v_i ≥ 0
```

これを **Charnes-Cooper 変換** で線形化すると、入力指向の主問題が得られる:

```
(CCR_in, primal)
max  Σ_r u_r y_{r0}
s.t. Σ_i v_i x_{i0} = 1
     Σ_r u_r y_{rj} - Σ_i v_i x_{ij} ≤ 0   ∀ j
     u_r, v_i ≥ 0
```

その双対 (envelope form):

```
(CCR_in, dual)
min  θ
s.t. Σ_j λ_j x_{ij} ≤ θ x_{i0}    ∀ i
     Σ_j λ_j y_{rj} ≥ y_{r0}      ∀ r
     λ_j ≥ 0
```

`θ* ∈ (0, 1]` が CCR 効率性。`θ* = 1` かつ全スラックがゼロのとき**完全効率**である。

**性質:**
- スケール不変 (CRS なので)
- ピア集合 `E_0 = { j | λ_j^* > 0 }` が DMU 0 の参照点
- 最適解は退化しうる (複数最適のスラック問題 → 第2段階 LP で対処)

### 3.3 BCC モデル (VRS)

`Σ_j λ_j = 1` を双対に追加することで凸結合制約を課し、VRS フロンティアを得る:

```
(BCC_in, dual)
min  θ
s.t. Σ_j λ_j x_{ij} ≤ θ x_{i0}    ∀ i
     Σ_j λ_j y_{rj} ≥ y_{r0}      ∀ r
     Σ_j λ_j = 1
     λ_j ≥ 0
```

主問題には自由変数 `u_0` (規模変数) が登場し、規模に関する性質を判定できる:

| 主問題の `u_0*` | 規模性質 |
|------|---------|
| `u_0* > 0` | DRS (規模に関する収穫逓減) |
| `u_0* = 0` | CRS |
| `u_0* < 0` | IRS (規模に関する収穫逓増) |

### 3.4 産出指向と双方向

**産出指向** (output-oriented) は `max φ` の形を取り、`φ* ≥ 1` を得る。
CRS では `θ_in^{CCR} = 1 / φ_out^{CCR}` だが、VRS では一般に成り立たない。

**双方向 (hyperbolic)** は `max φ` と `min θ` を同時に行うモデル (Färe et al. 1985)。

### 3.5 加法モデル (Additive Model)

**Charnes et al. (1985)** は radial efficiency の問題点（スラックを完全に捉えられない）を回避する加法モデルを提案:

```
max  Σ_i s_i^- + Σ_r s_r^+
s.t. Σ_j λ_j x_{ij} + s_i^- = x_{i0}    ∀ i
     Σ_j λ_j y_{rj} - s_r^+ = y_{r0}    ∀ r
     Σ_j λ_j = 1, λ, s ≥ 0
```

加法モデルは効率性を `[0, 1]` に正規化しないが、**Pareto-Koopmans 効率性**を直接判定できる。

### 3.6 SBM: Slacks-Based Measure

**Tone (2001, EJOR)** はスラック由来の非効率性を直接スコア化した:

```
ρ = (1 - (1/p) Σ_i s_i^- / x_{i0}) / (1 + (1/q) Σ_r s_r^+ / y_{r0})
```

`ρ ∈ (0, 1]`、ゼロ点は除外。SBM はラジアル効率性とは異なり**スラックに対して単調**である。

**変種:**
- SBM-CRS / SBM-VRS
- Super-SBM (Tone 2002)
- Network SBM (Tone, Tsutsui 2009)
- Non-radial / non-oriented family

### 3.7 スーパー効率性

**Andersen, Petersen (1993, MS)** は効率な DMU 間にも順位付けを行うため、対象 DMU を参照集合から除外する:

```
(Super-CCR_in)
min  θ
s.t. Σ_{j ≠ 0} λ_j x_{ij} ≤ θ x_{i0}
     Σ_{j ≠ 0} λ_j y_{rj} ≥ y_{r0}
     λ ≥ 0
```

`θ*` は 1 を超えうる。`θ* > 1` の DMU はそれだけ堅牢な効率性を持つ。
ただし VRS では実行不可能になることがあり、SBM-Super (Tone 2002) で安定化される。

### 3.8 クロス効率性

**Sexton, Silkman, Hogan (1986)** は各 DMU の最適重みを他 DMU に適用したクロス評価を提案。
クロス効率性行列 `E_{kj} = (Σ u_r^k y_{rj}) / (Σ v_i^k x_{ij})` の列平均で順位を確定する。
Aggressive / Benevolent の二次目的で重みの非一意性を解消する。

### 3.9 ネットワーク DEA / 二段階 DEA

DMU 内部に複数のサブプロセスがあり、中間財 `z` で接続される構造を扱う。

```
(Stage 1)         (Stage 2)
 x  →  [ proc1 ]  →  z  →  [ proc2 ]  →  y
```

**Färe, Grosskopf (2000)** や **Tone, Tsutsui (2009)** などが系統化。
中間財の値を内生変数として扱うことで「ブラックボックス DEA」を解消する。

### 3.10 ブートストラップ DEA (Simar-Wilson)

決定論的 DEA は信頼区間を持たないが、**Simar, Wilson (1998, 2000)** はサンプリング分布をブートストラップで近似した。

**手順 (smooth bootstrap):**
1. `θ̂_j` を全 DMU に対して計算
2. カーネル平滑化により分布 `f̂(θ)` を構成
3. ブートストラップ標本 `θ_j^*` を生成し、対応する擬似データ `(x_j^*, y_j^*)` を構成
4. ブートストラップサンプルで DEA を再実行 → `θ̂_j^*`
5. バイアス補正・信頼区間を計算

**Simar, Wilson (2007)** は二段階 DEA + truncated regression での外生変数効果分析を提案。
標準的な Tobit 回帰には統計的問題があり、truncated bootstrap が推奨される。

### 3.11 望ましくない産出を含む DEA

§9 を参照。方向距離関数や hyperbolic オリエンテーションを併用する。

> **実装リンク (将来):** `src/ml_sfa/frontier/dea.py`、`tests/integration/test_dea_*.py` を予定。

---

## 4. FDH: Free Disposal Hull とロバストフロンティア

### 4.1 FDH の動機: 凸性仮定の緩和

DEA は T の凸性を仮定する。しかし生産集合が凸でない場合 (規模・範囲の経済性、整数性、技術選択の不連続性)、
DEA は **真のフロンティアを下方バイアスする**。
**Deprins, Simar, Tulkens (1984)** は凸結合を許容しない FDH を提案した。

### 4.2 FDH の定式化

`λ_j ∈ {0, 1}` の二値制約と `Σ λ_j = 1` を課す:

```
(FDH_in)
min  θ
s.t. Σ_j λ_j x_{ij} ≤ θ x_{i0}      ∀ i
     Σ_j λ_j y_{rj} ≥ y_{r0}        ∀ r
     Σ_j λ_j = 1
     λ_j ∈ {0, 1}
```

これは整数計画だが、**FDH のラジアル効率性は単純比較で計算できる**:

```
θ_FDH(x_0, y_0) = min_{ j : y_j ≥ y_0 }  max_i (x_{ij} / x_{i0})
```

つまり、産出が DMU 0 を支配 (`y_j ≥ y_0`) する全 DMU に対して、入力比率の最大値を取り、その最小値を採る。
`O(n)` 時間で計算可能で、LP を必要としない。

### 4.3 ロバストフロンティア (Order-m, Order-α)

FDH も外れ値・極端 DMU の影響を受ける。
**Cazals, Florens, Simar (2002, J. Econometrics)** は **Order-m** を導入した:

```
γ_m(y) = E[ min_{ k = 1, ..., m } max_i (X_i^k / x_{i,0}) | Y^k ≥ y ]
```

産出が `y_0` を支配する母集団から `m` 個をブートストラップ抽出し、その最良点との比較を取る。
`m → ∞` で FDH に収束する。

**Aragon, Daouia, Thomas-Agnan (2005)** は **Order-α** 分位点フロンティアを提案:

```
γ_α(y) = inf { x | F_{X | Y ≥ y}(x) ≥ α }
```

`α = 1` で FDH、`α < 1` で外れ値に頑健なフロンティア。
収束率と漸近正規性が証明されており、ノンパラ統計の基礎を持つ。

### 4.4 FDH と DEA の関係

```
T_FDH ⊆ T_DEA-VRS ⊆ T_DEA-CRS
```

したがって `θ_FDH ≥ θ_BCC ≥ θ_CCR` が常に成立する。
FDH は最も保守的（効率性スコアが最も高く出やすい）な決定論的フロンティアである。

> **実装リンク (将来):** `src/ml_sfa/frontier/fdh.py`、`order_m.py`、`order_alpha.py`。

---

## 5. 半パラメトリック・決定論的手法 (COLS / MOLS / TFA / DFA)

### 5.0 Aigner-Chu (1968) 線形計画フロンティア

SFA 以前の決定論的パラメトリック手法。Cobb-Douglas 関数 `ln y = β_0 + Σ β_j ln x_j - u` で、
`u ≥ 0` をすべての観測に課して LP / QP で `β` を推定する。
SFA の確率成分 `v` を持たず、外れ値の影響を強く受ける。

### 5.1 COLS (Corrected OLS)

**Winsten (1957)** が提案した最古のフロンティア推定法のひとつ。

**手順:**
1. OLS で `ln y = β_0 + Σ β_j ln x_j + ε` を推定
2. 残差 `ε̂_i` の最大値で切片を補正: `β̂_0^c = β̂_0 + max_i ε̂_i`
3. 補正残差 `u_i = max_j ε̂_j - ε̂_i ≥ 0` を非効率性とみなす

**性質:**
- フロンティアは平行シフトのみ → 関数形が誤特定されると深刻なバイアス
- 全観測中の最良 DMU が完全効率 (TE = 1)
- ノイズと非効率性の分離なし

### 5.2 MOLS (Modified OLS)

COLS の改良として **Richmond (1974)** や Greene (1980) は分布仮定を加えた補正を提案:

```
β̂_0^m = β̂_0 + E[u]   (例: half-normal なら σ_u √(2/π))
```

非効率性分布のモーメントから期待値を計算して切片を補正する。
SFA との中間手法だが、ノイズ `v` を含めない点で決定論的フロンティアの一種である。

### 5.3 Thick Frontier Approach (TFA)

**Berger, Humphrey (1991, 1992)** が金融機関の効率性研究のために提案。

**手順:**
1. DMU を四分位等で層別
2. 最効率な層 (上位四分位) のみで OLS フロンティアを推定
3. 最非効率層 (下位四分位) との差を非効率性とみなす

層内のばらつきはノイズ、層間の差は非効率性とみなす近似。
分布仮定を回避するが、層の閾値選択に恣意性が残る。

### 5.4 DFA: Distribution-Free Approach

**Berger (1993)** はパネルデータでの個別効率性推定のため、時間平均を利用:

```
y_{it} = f(x_{it}; β) + α_i + v_{it}
```

`v` の時間平均はゼロに収束 (大数の法則) するため、`α̂_i = mean_t (y_{it} - f̂(x_{it}))` が
個別効率性の一致推定量となる。固定効果モデルの効率性解釈版である。

### 5.5 これらの位置付け

COLS / MOLS / TFA / DFA は以下の特徴を共有する:

- パラメトリック関数形を仮定 (関数形誤特定リスク)
- 計算が極めて軽量 (LP / OLS の組合せ)
- ノイズと非効率性の分離は弱い
- 銀行・保険など大規模パネルでベンチマーク的に多用される

> **実装リンク (将来):** `src/ml_sfa/frontier/cols.py`、`tfa.py`、`dfa.py`。

---

## 6. StoNED と CNLS: DEA と SFA の橋渡し

### 6.1 CNLS: Convex Nonparametric Least Squares

**Hildreth (1954)** に淵源を持ち、**Kuosmanen (2008)** が SFA 文脈で再活性化した。
凸 (concave) かつ単調な関数族の中で残差二乗和を最小化する:

```
(CNLS, 単一産出, 凹単調仮定)
min  Σ_i (y_i - α_i - β_i' x_i)²
s.t. α_i + β_i' x_i ≤ α_j + β_j' x_i    ∀ i, j   (Afriat 不等式)
     β_i ≥ 0                              ∀ i
```

各観測 `i` ごとに局所線形支持平面 `(α_i, β_i)` を持ち、
**Afriat (1972) 不等式**で凹性を強制する点が要諦。
解は QP で得られる凹単調区分線形関数となる。

**性質:**
- 関数形を一切仮定しない
- 凹単調性を厳密に保証
- 解の局所線形構造により限界生産性が観測単位で得られる
- `O(n²)` 制約で大規模時の計算は QP ソルバ依存

### 6.2 StoNED: Stochastic Non-smooth Envelopment of Data

**Kuosmanen, Kortelainen (2012, J. Productivity Analysis)** は CNLS と SFA を組み合わせた:

**ステップ:**
1. CNLS で `f̂(x)` と複合誤差 `ε̂_i = y_i - f̂(x_i)` を推定
2. `ε̂` の歪度から `σ̂_u`, `σ̂_v` を Method of Moments で同定 (Olson et al. 1980 と同型)
3. 各観測の `u_i` を Jondrow et al. (1982) の条件付き期待値で推定

これにより、

- フロンティア関数形仮定を排除 (CNLS で凸単調にのみ縛る)
- 確率誤差 `v` を保持 (DEA / FDH に対する優位)
- 非効率性分布仮定 (half-normal 等) は依然必要

DEA の柔軟性と SFA の確率モデルを統合した代表的手法であり、`pyStoNED` に実装がある。

### 6.3 関連手法

- **Sign-constrained CNLS:** Kuosmanen & Johnson (2010)。回帰係数の符号制約のみで凸性を緩和。
- **Stochastic Semi-Nonparametric Frontier:** Park, Simar, Zelenyuk (2015)。
- **Local linear smoothing of CNLS:** 平滑化バンド幅の選択でバイアス・分散を制御。

> **実装リンク (将来):** `src/ml_sfa/frontier/cnls.py`、`stoned.py`。

---

## 7. ML 系ノンパラメトリック手法 (EAT / RF / Boosting)

### 7.1 EAT: Efficiency Analysis Trees

**Esteve, Aparicio, Rabasa, Rodriguez-Sala (2020, Expert Systems with Applications)** は CART を効率性分析に拡張した。

**特徴:**
- 産出を推定する回帰木を学習
- 葉ノードの予測値を**フリーディスポーザリティ性を保つように**修正
- FDH の単調性を保ったまま、有限サンプルでの推定を安定化

**アルゴリズム要旨:**

```
1. 通常の CART 回帰木で y を学習
2. 葉ノードの推定値に対し、入力空間で支配関係にある葉同士で max を取る
3. 結果として得られる予測関数は単調かつ階段状 (FDH と同じ離散性)
```

**RFEAT (Random Forest EAT):** 同著者 (2023)。Bagging で EAT をアンサンブル化。
Order-m と類似のロバスト化効果を持つ。

### 7.2 ノンパラメトリック回帰系の応用

| 手法 | フロンティア化のための制約 |
|------|---------------------|
| Random Forest | 単調性は保証されない → 後処理で max-aggregation |
| Gradient Boosting | 単調制約を持つ実装 (LightGBM monotone_constraints) を使用 |
| Kernel regression (Nadaraya-Watson) | 局所最適化で単調化、または等濃度バンド幅 |
| Spline (B-spline, P-spline) | shape constraint (Pya & Wood 2015) |

これらは「決定論的フロンティア」の枠で SFA の確率成分を持たないが、
shape 制約を組合せて DEA / FDH の代替を成す。

### 7.3 Robust Nonparametric SFA (Zheng et al. 2024)

**Zheng et al. (arXiv:2404.04301, 2024)** は外れ値耐性のあるノンパラメトリック SFA を提案。
quantile loss + shape constraint で robust frontier を推定し、CNLS を Huber 化する。

> **実装リンク (将来):** `src/ml_sfa/frontier/eat.py`、`rfeat.py`。

---

## 8. 動学・生産性指数

### 8.1 Malmquist 生産性指数 (MPI)

**Caves, Christensen, Diewert (1982)** が定義し、**Färe et al. (1992)** が DEA で計算可能にした。

時点 `t` と `t+1` の DMU について、4 つの距離関数を計算:

```
M_O(t, t+1) = [ (D^t_O(x^{t+1}, y^{t+1}) / D^t_O(x^t, y^t))
              × (D^{t+1}_O(x^{t+1}, y^{t+1}) / D^{t+1}_O(x^t, y^t)) ]^{1/2}
```

幾何平均で**基準時点バイアス**を回避。MPI は次のように分解できる:

```
MPI = TE_change × Frontier_shift
    = (D^{t+1}_O(x^{t+1}, y^{t+1}) / D^t_O(x^t, y^t))
    × [ (D^t_O(x^{t+1}, y^{t+1}) / D^{t+1}_O(x^{t+1}, y^{t+1}))
       × (D^t_O(x^t, y^t) / D^{t+1}_O(x^t, y^t)) ]^{1/2}
```

第一項が**追いつき効果**、第二項が**技術前進効果**。

**注意 (Pastor, Lovell 2005):** MPI は循環性を持たないなどの欠点がある。代替として:

- **Hicks-Moorsteen 指数** (Bjurek 1996): 入力・産出双方を独立に評価
- **Färe-Primont 指数** (O'Donnell 2014): 完全な指数性質を満たす transitive 指数

### 8.2 Luenberger 生産性指標

DDF を用いた加法版生産性指標。望ましくない産出を扱える点で MPI より柔軟。

> **実装リンク (将来):** `src/ml_sfa/frontier/productivity.py`。

---

## 9. 望ましくない産出と方向距離関数

### 9.1 問題設定

CO2 排出、不良品、不良債権などの**望ましくない産出 (undesirable output)** を含む生産では、
標準 DEA は不適切である ((y, b) のうち `b` を増やすことが効率改善とみなされる病理が起こる)。

**Färe, Grosskopf, Lovell, Pasurka (1989)** は弱処分性 (weak disposability) を導入し、
望ましい産出 `y` と望ましくない産出 `b` の同時拡張・縮減を要求した。

### 9.2 方向距離関数 (DDF) による定式化

方向ベクトル `g = (-g_x, g_y, -g_b)` を取り:

```
β*(x_0, y_0, b_0) = max β
s.t. Σ_j λ_j x_{ij} ≤ x_{i0} - β g_x^i      ∀ i
     Σ_j λ_j y_{rj} ≥ y_{r0} + β g_y^r      ∀ r
     Σ_j λ_j b_{kj} = b_{k0} - β g_b^k      ∀ k    (= で弱処分)
     Σ_j λ_j = 1, λ ≥ 0
```

`β* = 0` で効率的、`β* > 0` で非効率な分だけ望ましい産出を増やしつつ望ましくない産出を減らせる。

### 9.3 Slacks-Based Measure for Undesirable Outputs

Tone (2003), Cooper et al. (2007) などが SBM を `b` に拡張:

```
ρ = (1 - (1/p) Σ_i s_i^- / x_{i0})
    / (1 + (1/(q+r)) [Σ y s_y^+ / y_{r0} + Σ b s_b^- / b_{k0}])
```

### 9.4 環境効率性ベンチマーク

電力・製造業における CO2 排出効率分析、銀行の不良債権処理効率など、
方向距離関数とそのMLサロゲート (e.g., Färe-Primont with bad outputs) は応用が広い。

> **実装リンク (将来):** `src/ml_sfa/frontier/ddf.py`、`undesirable.py`。

---

## 10. 手法間の体系的比較

### 10.1 仮定とロバスト性

| 手法 | 凸性 | 単調性 | 確率誤差 | 外れ値感度 | 計算量 |
|------|:---:|:---:|:---:|:---:|:---|
| CCR (CRS) | ○ | ○ | × | 高 | LP × n |
| BCC (VRS) | ○ | ○ | × | 高 | LP × n |
| SBM | ○ | ○ | × | 高 | LP × n |
| FDH | × | ○ | × | 高 | O(n²) (比較) |
| Order-m | × | ○ | (準) | **低** | O(B n²) (B: ブートストラップ) |
| Order-α | × | ○ | (準) | **低** | O(n²) |
| COLS | (関数依存) | (関数依存) | × | 高 | OLS |
| StoNED | ○ (CNLS) | ○ | ○ | 中 | QP + MoM |
| CNLS | ○ | ○ | ○ | 中 | QP |
| EAT | × | ○ | × | 中 | 木学習 |
| RFEAT | × | ○ | × | **低** | O(B 木学習) |
| SFA (ALS) | (関数依存) | (関数依存) | ○ | 中 | MLE |

### 10.2 手法選択フローチャート

```
START
  │
  ├─ 確率誤差が必要か?
  │    │
  │    ├─ Yes ──┬─ 関数形仮定をしたい? ──→ SFA (ml_sfa_survey 参照)
  │    │         └─ ノンパラ + 確率誤差 ──→ StoNED / Robust nonparametric SFA
  │    │
  │    └─ No ───┬─ 凸性は妥当か?
  │             │    │
  │             │    ├─ Yes ──┬─ 規模仮定がある? → CCR (CRS)
  │             │    │         └─ なし          → BCC (VRS) / SBM
  │             │    │
  │             │    └─ No ───→ FDH
  │             │
  │             └─ 外れ値が懸念? ──→ Order-m / Order-α / RFEAT
  │
  └─ 動学パネル? ──→ Malmquist / Hicks-Moorsteen / Färe-Primont
```

### 10.3 結果の比較プロトコル

複数手法間で効率性スコアを比較する際の推奨手順:

1. **同一データに対して複数手法を適用**
2. **Spearman 順位相関**で効率性ランキングの整合性を検証
3. **Li (1996) test** または KS 検定で分布同等性をテスト
4. **ブートストラップ信頼区間**で点推定値の不確実性を提示
5. **条件付き効率性** (Daraio, Simar 2005) で外生変数の影響を補正

> **実装リンク (将来):** `src/ml_sfa/frontier/comparison.py`、`evaluation/frontier_metrics.py`。

---

## 11. 動作検証用データセット

### 11.1 検証データの三層構造

将来の単体・統合テストでは以下の三層を組み合わせて手法を検証する。
真値が既知なシミュレーション、公表値の再現、外部実装との数値同等性の三軸を確保する。

```
検証データ階層
├── L1: シミュレーション DGP   (再現可能、CI 完全自走、真値既知)
│       単位テスト・回帰テスト・収束率検証の主軸
│
├── L2: 同梱用ミニデータセット (公開済・小規模・再配布可能)
│       公表値との一致を統合テストで検証
│       他ソフトウェア (pyStoNED, R Benchmarking) との数値同等性
│
└── L3: 外部ダウンロード (オプション、CI 非依存)
        大規模・ライセンス制約のあるデータ
        論文用ベンチマーク・長期評価に使用
```

### 11.2 シミュレーション DGP (L1)

L1 は**真値 (true frontier, true TE)** を握るため、推定誤差を直接定量化できる。
本ライブラリの `simulate_sfa()` (`src/ml_sfa/data/simulator.py`) を起点に、決定論・ノンパラ系専用の DGP を追加する想定。

#### 11.2.1 Cobb-Douglas + half-normal (基本検証)

```
ln y_i = β_0 + Σ_j β_j ln x_{ij} + v_i - u_i
v_i ~ N(0, σ_v²),  u_i ~ |N(0, σ_u²)|
```

| 検証対象 | 真値 | 評価指標 |
|----------|------|----------|
| ParametricSFA | β, σ_v, σ_u, TE_i | RMSE(β), RMSE(TE), 順位相関 |
| COLS / MOLS | β | 切片補正の漸近一致性 |
| CNLS / StoNED | f, σ_u, σ_v | フロンティア L2 誤差 |
| DEA-CCR | TE (CRS の場合) | 順位相関、収束率 |

既存の `ml_sfa.data.simulator` でカバーされる。決定論手法でも基準として併用する。

#### 11.2.2 Banker-Natarajan (2008) DGP — DEA 検証

Banker, Natarajan (2008) "Evaluating Contextual Variables Affecting Productivity Using DEA" *Operations Research* に従う:

```
y_i = f(x_i) - u_i,  u_i ≥ 0
f(x) = x_1^{0.4} x_2^{0.4}   (凹単調 Cobb-Douglas, CRS 検証時は β_1 + β_2 = 1)
```

DEA 効率性の収束率は `n^{-2 / (p + q + 1)}` (Korostelev et al. 1995, Kneip et al. 2008)。
サンプルサイズを変動させて経験的に確認する。

#### 11.2.3 Simar-Wilson DGP — ブートストラップ DEA 検証

Simar, Wilson (1998, 2007) のシミュレーション設計:

```
x_i ~ U(1, 4),  y_i = x_i^{0.5} · exp(-u_i),  u_i ~ Exp(1/3)
```

- ブートストラップ信頼区間の経験的カバー率 (期待値 0.95)
- truncated bootstrap (Simar-Wilson 2007) の二段階回帰で外生変数効果のバイアス補正検証

#### 11.2.4 Cazals-Florens-Simar DGP — Order-m / Order-α 検証

汚染分布混合モデル:

```
(x_i, y_i) ~ (1 - ε) · F_clean + ε · F_outlier
```

外れ値比率 `ε ∈ {0, 0.05, 0.10}`、`m` の選択による Bias-Variance トレードオフを検証。
DEA / FDH の効率スコアが外れ値で激しく変動する一方、Order-m / Order-α は安定であることを示す。

#### 11.2.5 非凸 DGP — FDH 検証

階段状フロンティア (規模の不経済を持つ生産集合):

```
f(x) = floor(x · 5) / 5    (区分定数で非凸)
```

DEA-VRS は凸結合で内挿し下方バイアスを示す一方、FDH は階段を保つ。
両手法のスコア差が真の非凸性に対応することを検証。

#### 11.2.6 望ましくない産出 DGP — DDF / SBM 検証

(y, b) 同時生成:

```
y_i = f(x_i) - u_i,  b_i = g(x_i, y_i) + η_i
g(x, y) = c · y^{α}    (環境 Kuznets 型)
```

弱処分性 (`y` と `b` の同時拡張・縮減) を満たす集合で DDF 値を計算し、
方向ベクトル `g` に対する scaling 性質を検証。

#### 11.2.7 パネル DGP — Malmquist 検証

時変フロンティアと時変効率性の同時変動:

```
f_t(x) = A_t · x^β,  A_{t+1} = A_t · (1 + γ)   (技術進歩 γ)
TE_{i,t} = exp(-u_{i,t}),  u_{i,t} = ρ u_{i,t-1} + ξ_t  (Battese-Coelli 1995)
```

Malmquist 指数の TE_change × Frontier_shift 分解が真値と一致するかを検証。

### 11.3 公開データセット (L2)

公表論文で効率性スコアが open に提示されており、再配布可能 / 広く流通している小規模データセット。
**「公表値との数値一致」を統合テストの主指標**にできる。

| # | データセット | n × T | 投入 | 産出 | 出典 (公表値あり) | 検証対象 | 取得元 |
|---|-------------|-------|-----|-----|----------------|----------|--------|
| D1 | Charnes-Cooper-Rhodes (1981) PFT | 70 | 5 | 3 | Charnes, Cooper, Rhodes (1981, *MS*) Table 1 | DEA-CCR (基本) | R `Benchmarking::charnes1981` |
| D2 | Banker-Charnes-Cooper (1984) hospitals | 14 | 2 | 2 | Banker, Charnes, Cooper (1984, *MS*) Table 1 | DEA-BCC, scale eff | 論文付録 |
| D3 | Christensen-Greene (1976) electricity | 123 | 4 | 1 | Greene (2003 textbook) Table F4.4 | SFA, COLS, ベンチマーク | Greene 教科書付録 / R `frontier::electricity` |
| D4 | Coelli (1996) Philippine rice | 43 × 8 | 4 | 1 | Coelli (1996) CEPA WP | パネル SFA, Malmquist | R `frontier::riceProdPhil` |
| D5 | Tulkens (1993) Belgian post offices | 789 | 1 | 1 | Tulkens (1993, *JPA*) | FDH (オリジナル例) | 論文公開 |
| D6 | Färe et al. (1992) Swedish pharmacies | 42 × 9 | 3 | 4 | Färe et al. (1992, *JPA*) | Malmquist | 論文付録 |
| D7 | Tone (2001) SBM example | 12 | 2 | 2 | Tone (2001, *EJOR*) Table 1 | SBM | 論文 / `pyDEA` examples |
| D8 | Aigner-Lovell-Schmidt (1977) primary metals | ~50 | 2 | 1 | Aigner et al. (1977, *JoE*) | SFA 起源 | 論文付録 |
| D9 | Norwegian wood production (`norWood`) | 21 | 2 | 1 | Bogetoft, Otto (2011) 教科書 | DEA / FDH 教育用 | R `Benchmarking::norWood` |
| D10 | Milk production (`milkProd`) | 108 | 4 | 1 | Bogetoft, Otto (2011) 教科書 | DEA-VRS, ブートストラップ | R `Benchmarking::milkProd` |
| D11 | Battese-Coelli (1992) Indian rice | 38 × 10 | 4 | 1 | Battese, Coelli (1992, *JPA*) | 時変 SFA, パネル | 論文付録 |
| D12 | Banks-20 (`banks20`) | 20 | 3 | 3 | Bogetoft, Otto (2011) 教科書 | DEA, クロス効率 | R `Benchmarking::banks20` |
| D13 | Pig farms (`pigData`) | 50 | 5 | 1 | Bogetoft, Otto (2011) 教科書 | DEA-VRS | R `Benchmarking::pigData` |
| D14 | Finnish electricity distribution (StoNED) | ~100 | 3 | 2 | Kuosmanen (2012, EI experiment) | StoNED, CNLS | `pyStoNED` 同梱 |

**取得・再配布の方針:**
1. R Benchmarking / R frontier / pyStoNED から CSV エクスポートし `tests/data/published/<dataset>.csv` に同梱
2. 各 CSV と同階層に出典・ライセンス・公表値テーブルを記した `<dataset>.md` を配置
3. データ容量が大きい場合 (D5 の Belgian post offices 789 obs など) はサブセット版 + フル版ダウンロードスクリプトの併用

### 11.4 既存パッケージ同梱データの再利用

公表データを実装ごとに収集する代わりに、確立済みパッケージのデータをそのまま流用する戦略。
ライセンス確認の上で `tests/data/published/` に CSV として固定スナップショットを保存する。

| ソース | 形式 | 含まれる代表データ | ライセンス |
|--------|------|------------------|------------|
| R `Benchmarking` (Bogetoft, Otto) | `.RData` | charnes1981, milkProd, banks20, norWood, pigData, frontier71 | GPL-2 (再配布要注意) |
| R `frontier` (Coelli, Henningsen) | `.RData` | electricity, riceProdPhil, front41Data | GPL-2 |
| R `deaR` | `.rda` | Hospitals_2017, Universities_2017 | GPL-3 |
| R `eat` (Esteve et al.) | `.rda` | PISAindex, EAT 教育用合成データ | GPL-3 |
| Python `pyStoNED` (Dai, Kuosmanen) | CSV + Python | Finnish DSO, Korean banks 等 | MIT |
| Python `pyDEA` | CSV | 教科書例題 (Cooper et al. 2007) | MIT |

**GPL データ流用時の注意:** GPL-2/3 のデータを CSV としてリポジトリに同梱する場合、
本プロジェクトの MIT との互換性整理が必要。**Proposal 経由で確定すること** (`change-gate.md` 参照)。
代替案: テスト時にダウンロードする `tests/data/external/download.py` を用意し、リポジトリには含めない。

### 11.5 大規模・外部データセット (L3, オプション)

CI には組み込まず、論文用ベンチマークや長期検証で使用する。

| データセット | 規模 | 提供元 | 想定検証 |
|-------------|------|--------|----------|
| EU KLEMS productivity | 多国 × 多年 × 多産業 | euklems-intanprod-llee.luiss.it | クロスカントリー Malmquist |
| WORLD KLEMS | 同上 | worldklems.net | 同上 |
| US BLS productivity statistics | 産業別パネル | bls.gov | 動学生産性指数 |
| IEA electricity / energy efficiency | 国別パネル | iea.org | 望ましくない産出 (CO2) |
| World Bank WDI | 国別マクロ | data.worldbank.org | スケール効果検証 |
| Federal Reserve Y9-C banking | 米銀パネル | ffiec.gov | 銀行効率 (DFA, TFA) |
| Penn World Tables 10.0 | 国別マクロ | rug.nl/ggdc/productivity/pwt | 全要素生産性 |

これらは API / FTP からの取得が必要で、ライセンス・利用規約・データクリーニングが手法検証本体に対して費用が大きい。
**統合テストに必須としない** が、研究応用例ノートブック (`notebooks/`) で利用する想定。

### 11.6 データ × 手法 対応マトリクス

各データセットがどの手法のテストに使えるかを整理する。
記号: ◎ = 公表値あり (一致確認可能) / ○ = 適用可能 / △ = サブセット適用 / × = 不適。

| 手法 \\ Dataset | DGP-1 (CD) | DGP-2 (BN08) | DGP-3 (SW) | DGP-4 (CFS) | DGP-5 (非凸) | D1 PFT | D2 Hosp | D3 Elec | D4 Rice | D5 Post | D6 Pharm | D7 SBM | D9-13 (R Bench) | D14 Stoned |
|-----------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DEA-CCR (§3.2) | ○ | ◎ | ○ | △ | × | ◎ | ○ | × | × | △ | × | ○ | ◎ | ○ |
| DEA-BCC (§3.3) | ○ | ◎ | ○ | △ | × | ○ | ◎ | × | × | △ | × | ○ | ◎ | ○ |
| SBM (§3.6) | ○ | ○ | ○ | △ | × | ○ | ○ | × | × | × | × | ◎ | ○ | ○ |
| Super-eff (§3.7) | ○ | ○ | ○ | △ | × | ○ | ○ | × | × | × | × | ○ | ○ | ○ |
| Cross-eff (§3.8) | ○ | ○ | ○ | × | × | ○ | ○ | × | × | × | × | ○ | ○ | ○ |
| Bootstrap DEA (§3.10) | × | ○ | ◎ | ○ | × | ○ | ○ | × | × | × | × | × | ○ | ○ |
| FDH (§4.2) | ○ | ○ | ○ | △ | ◎ | ○ | ○ | × | × | ◎ | × | ○ | ○ | ○ |
| Order-m (§4.3) | × | ○ | ○ | ◎ | × | × | × | × | × | ○ | × | × | ○ | ○ |
| Order-α (§4.3) | × | ○ | ○ | ◎ | × | × | × | × | × | ○ | × | × | ○ | ○ |
| COLS (§5.1) | ◎ | × | × | × | × | × | × | ○ | ○ | × | × | × | × | × |
| StoNED (§6.2) | ◎ | ○ | × | × | × | × | × | ○ | ○ | × | × | × | × | ◎ |
| CNLS (§6.1) | ◎ | ○ | × | × | × | × | × | ○ | ○ | × | × | × | × | ◎ |
| EAT / RFEAT (§7.1) | ○ | ○ | × | ○ | ◎ | ○ | × | × | × | × | × | × | ○ | × |
| Malmquist (§8.1) | × | × | × | × | × | × | × | × | ◎ | × | ◎ | × | × | × |
| DDF (§9.2) | △ DGP-6 | × | × | × | × | × | × | × | × | × | × | × | × | × |

**読み方:**
- 列がデータセット、行が手法
- ◎ がついているセルは「一致テスト」(reproduce published values) を構成可能
- ○ は「動作テスト」(estimator runs without error, basic property checks)
- 空 / × は不適 (例: 単一産出データに SBM-multi-output は不適)

### 11.7 検証プロトコル

#### 11.7.1 単体テスト (L1: シミュレーション)

```python
# tests/unit/frontier/test_<method>_simulation.py の概形
def test_dea_ccr_consistency_under_banker_natarajan_dgp():
    rng = np.random.default_rng(42)
    n_grid = [50, 100, 200, 400]
    rmse_grid = []
    for n in n_grid:
        data = simulate_banker_natarajan(n=n, p=2, rng=rng)
        model = DEA(orientation="input", returns="crs").fit(data.X, data.Y)
        te_hat = model.efficiency(data.X, data.Y)
        rmse_grid.append(rmse(te_hat, data.true_te))
    # 収束率 n^{-2/(p+q+1)} の経験的確認
    assert rmse_grid[-1] < rmse_grid[0] * 0.5
```

| テスト種別 | 検証内容 | 合否基準 (例) |
|----------|---------|---------------|
| バイアス | n→∞ で 0 に収束 | n=400 で `|bias| < 0.05` |
| RMSE | 既知レートで減衰 | 倍数 n で RMSE が `2^{-2/(p+q+1)}` 倍以下 |
| 順位相関 | 効率性ランキング | Spearman ρ ≥ 0.85 (n=200) |
| カバー率 | ブートストラップ CI | 経験的 cov ∈ [0.92, 0.98] (B=2000) |
| 単調性 | フロンティアの単調性 | 数値微分の最小値 ≥ -ε |
| スケール不変 | DEA-CCR の単位独立性 | スコア差 < 1e-9 |

#### 11.7.2 統合テスト (L2: 公開データ)

```python
# tests/integration/frontier/test_charnes1981_reproduction.py
def test_ccr_reproduces_published_efficiencies():
    df = pd.read_csv("tests/data/published/charnes1981.csv")
    model = DEA(orientation="input", returns="crs").fit(df[INPUTS], df[OUTPUTS])
    te = model.efficiency(df[INPUTS], df[OUTPUTS])
    published = pd.read_csv("tests/data/published/charnes1981_published_te.csv")
    np.testing.assert_allclose(te, published["te"].values, atol=1e-3)
```

検証対象:
- **公表値再現**: 論文 / 教科書のスコアと数値一致 (許容誤差 1e-3 ~ 1e-2)
- **クロスソフトウェア整合**: pyStoNED / R `Benchmarking` の結果と数値一致
- **ピア集合の一致**: ピア (`λ_j > 0` の集合) が公表値と一致
- **規模性質判定**: BCC モデルの IRS / CRS / DRS 判定が一致

#### 11.7.3 プロパティベーステスト (Hypothesis 等)

| プロパティ | 対象 | テスト形 |
|----------|------|---------|
| 単位独立性 (units invariance) | DEA-CCR, BCC, SBM | 任意の正の対角スケールで効率性不変 |
| 並進不変性 (translation invariance) | Additive, SBM | 入力 / 産出への正の定数加算で効率性不変 |
| 単調性 | DEA, FDH, CNLS | x が増えると効率性は (弱) 減少しうる |
| 収束順序 | FDH ≤ BCC ≤ CCR | 同データで `θ_FDH ≥ θ_BCC ≥ θ_CCR` |
| 双対関係 | DEA primal/dual | 主問題と双対問題の最適値が一致 |

#### 11.7.4 オプショナル依存・スキップ制御

- L2/L3 のうち外部パッケージ参照テストは `pytest.importorskip("pystoned")` 等でスキップ
- L1 シミュレーションは依存ゼロで CI 必須
- 大規模データのテストは `@pytest.mark.slow` で分離、`make ci` の通常実行から除外

### 11.8 同梱予定のディレクトリ構成 (案)

```
tests/data/
├── published/                       # L2 公開データ (再配布可能なもの)
│   ├── charnes1981/
│   │   ├── data.csv
│   │   ├── published_te.csv         # 公表効率性スコア
│   │   └── REFERENCE.md             # 出典・ライセンス・列定義
│   ├── christensen_greene_1976/
│   │   ├── electricity.csv
│   │   ├── published_te.csv
│   │   └── REFERENCE.md
│   ├── tone_2001_sbm/
│   ├── coelli_1996_rice/
│   ├── ...
│   └── INDEX.md                     # 全データセットの目録
│
├── external/                        # L3 オンデマンドダウンロード
│   ├── download.py                  # 各データの取得スクリプト
│   ├── checksums.txt                # SHA-256 ハッシュ
│   └── .gitignore                   # 実データは git 管理外
│
└── simulation/                      # L1 シミュレーション (オプションでキャッシュ)
    └── (ランタイム生成、git 管理外)
```

### 11.9 取得・ライセンスのポリシー

| 種別 | 同梱可否 | 根拠 |
|------|:------:|------|
| 公表論文の Table データ (D1, D2, D7) | ○ (出典明記の上で) | 学術引用慣例、データ点数が小さい |
| Greene 教科書付録 (D3) | ○ | Greene が公開、教育目的 |
| R `frontier` 同梱データ (GPL-2) | △ | ライセンス整理が必要 |
| R `Benchmarking` 同梱データ (GPL-2) | △ | 同上、CSV 形式変換時の権利検討 |
| pyStoNED 同梱 CSV (MIT) | ○ | MIT 互換 |
| 政府統計 (BLS, Eurostat 等) | △ | データ提供条件確認 |
| 民間プロプライエタリ | × | 同梱不可 |

**運用方針:** GPL-2/3 データの直接同梱は避け、`tests/data/external/download.py` でユーザ環境に取得する形を基本とする。
公表論文 Table データは小規模で出典明記すれば実質問題ない。
個別データの判断は **HISTORY.md の Proposal** を介して確定する。

### 11.10 検証結果の集約

複数手法・複数データを横断する集約レポートを `notebooks/` 配下に置く想定:

```
notebooks/
├── frontier_validation/
│   ├── 01_dea_simulation_consistency.ipynb     # L1 検証
│   ├── 02_charnes1981_reproduction.ipynb       # L2 公表値再現
│   ├── 03_robust_frontier_outliers.ipynb       # Order-m 外れ値感度
│   ├── 04_stoned_vs_pystoned.ipynb             # クロスソフト検証
│   └── 05_method_comparison_overview.ipynb     # 全手法サマリ
└── ...
```

各ノートブックは「データ → 推定 → 真値/公表値との比較 → 図示」の標準パイプラインに従う。

> **実装リンク (将来):**
> - シミュレータ: 既存 `src/ml_sfa/data/simulator.py` を拡張、または `src/ml_sfa/frontier/simulators.py` を新設
> - 公開データローダ: `src/ml_sfa/data/published.py` (新設)
> - 検証ユーティリティ: `src/ml_sfa/evaluation/frontier_validation.py` (新設)

---

## 12. 本ライブラリにおける実装計画 (将来)

本サーベイは将来の実装と双方向リンクすることを意図している。
各手法に対し以下の対応が想定される (あくまで参考案 — 個別 Proposal 経由で確定する)。

### 12.1 ディレクトリ構成 (案)

```
src/ml_sfa/frontier/                # 決定論・ノンパラ系
├── __init__.py
├── _technology.py                   # 生産可能集合・距離関数の基底 (§2)
├── dea.py                           # CCR, BCC, additive (§3.2-3.5)
├── sbm.py                           # SBM, super-SBM (§3.6, 3.7)
├── cross_efficiency.py              # クロス効率性 (§3.8)
├── network_dea.py                   # ネットワーク・二段階 DEA (§3.9)
├── bootstrap_dea.py                 # Simar-Wilson ブートストラップ (§3.10)
├── fdh.py                           # FDH (§4.2)
├── order_m.py                       # Order-m (§4.3)
├── order_alpha.py                   # Order-α 分位点 (§4.3)
├── cols.py                          # COLS / MOLS (§5.1, 5.2)
├── tfa.py                           # Thick Frontier Approach (§5.3)
├── dfa.py                           # Distribution-Free Approach (§5.4)
├── cnls.py                          # Convex Nonparametric LS (§6.1)
├── stoned.py                        # StoNED (§6.2)
├── eat.py                           # Efficiency Analysis Trees (§7.1)
├── rfeat.py                         # Random Forest EAT (§7.1)
├── productivity.py                  # Malmquist / Hicks-Moorsteen (§8)
├── ddf.py                           # 方向距離関数 (§9.2)
└── undesirable.py                   # 望ましくない産出 SBM (§9.3)

tests/integration/frontier/
├── test_dea_ccr.py
├── test_dea_bcc.py
├── test_fdh.py
├── ...
```

### 12.2 共通インタフェース (案)

```python
class BaseFrontierEstimator(BaseEstimator):
    def fit(self, X, Y) -> Self: ...
    def efficiency(self, X, Y, *, orientation: Literal["input", "output"]) -> NDArray:
        ...
    def slacks(self, X, Y) -> tuple[NDArray, NDArray]:  # (s^-, s^+)
        ...
    def reference_set(self, X, Y) -> list[NDArray]:      # ピア
        ...
    def summary(self) -> FrontierSummary: ...
```

`ml_sfa.models.BaseSFAEstimator` (確率版) と並ぶ別系統として位置付け、
評価指標 (`evaluation/frontier_metrics.py`) では両系統を横断比較できるようにする。

### 12.3 Phase 計画 (案)

| Phase | 範囲 | 期待される成果物 |
|-------|------|-----------------|
| F1 | DEA-CCR / BCC / Additive / SBM | LP ソルバ統合、シミュレーション検証 |
| F2 | FDH + Order-m / Order-α | 非凸技術への対応、ロバストフロンティア |
| F3 | COLS / MOLS / TFA / DFA | パラメトリック決定論ベンチマーク |
| F4 | CNLS / StoNED | SFA との統合比較 |
| F5 | EAT / RFEAT | ML 系ノンパラ |
| F6 | Bootstrap DEA / Simar-Wilson 二段階 | 統計的推論 |
| F7 | Malmquist / 方向距離関数 / 望ましくない産出 | 動学・環境効率性 |

各 Phase は **HISTORY.md の Proposal** を経て確定する (ルール `change-gate.md` 参照)。
本ドキュメントはあくまで「網羅的サーベイ + 実装の地図」であり、実装そのものを規定しない。

### 12.4 実装と本ドキュメントの双方向リンク規約 (将来)

- 実装側: 各モジュールの docstring 冒頭で `See: docs/research/deterministic_frontier_survey.md#<anchor>` を明記
- 本ドキュメント: §12.1 の表に実装 PR / モジュールパスを追記
- テスト側: シミュレーション DGP の理論的根拠を本ドキュメントの定式化と相互参照

---

## 13. 参考文献

### 基礎・歴史
- Farrell, M. J. (1957). The Measurement of Productive Efficiency. *J. Royal Statistical Society A*, 120(3).
- Shephard, R. W. (1953). *Cost and Production Functions*. Princeton University Press.
- Aigner, D., Chu, S. F. (1968). On Estimating the Industry Production Function. *American Economic Review*, 58.

### DEA
- Charnes, A., Cooper, W. W., Rhodes, E. (1978). Measuring the Efficiency of Decision Making Units. *European Journal of Operational Research*, 2.
- Banker, R. D., Charnes, A., Cooper, W. W. (1984). Some Models for Estimating Technical and Scale Inefficiencies in DEA. *Management Science*, 30.
- Charnes, A., Cooper, W. W., Golany, B., Seiford, L., Stutz, J. (1985). Foundations of DEA for Pareto-Koopmans Efficient Empirical Production Functions. *J. Econometrics*, 30.
- Andersen, P., Petersen, N. C. (1993). A Procedure for Ranking Efficient Units in DEA. *Management Science*, 39.
- Sexton, T. R., Silkman, R. H., Hogan, A. J. (1986). Data Envelopment Analysis: Critique and Extensions. *New Directions for Program Evaluation*, 32.
- Tone, K. (2001). A Slacks-Based Measure of Efficiency in DEA. *European Journal of Operational Research*, 130.
- Tone, K. (2002). A Slacks-Based Measure of Super-Efficiency in DEA. *EJOR*, 143.
- Tone, K., Tsutsui, M. (2009). Network DEA: A Slacks-Based Measure Approach. *EJOR*, 197.

### FDH / ロバスト
- Deprins, D., Simar, L., Tulkens, H. (1984). Measuring Labor-Efficiency in Post Offices. In *The Performance of Public Enterprises*. North-Holland.
- Cazals, C., Florens, J.-P., Simar, L. (2002). Nonparametric Frontier Estimation: A Robust Approach. *J. Econometrics*, 106.
- Aragon, Y., Daouia, A., Thomas-Agnan, C. (2005). Nonparametric Frontier Estimation: A Conditional Quantile-based Approach. *Econometric Theory*, 21.

### COLS / MOLS / TFA / DFA
- Winsten, C. B. (1957). Discussion on Mr. Farrell's Paper. *J. Royal Statistical Society A*, 120.
- Richmond, J. (1974). Estimating the Efficiency of Production. *International Economic Review*, 15.
- Berger, A. N., Humphrey, D. B. (1991). The Dominance of Inefficiencies over Scale and Product Mix Economies in Banking. *J. Monetary Economics*, 28.
- Berger, A. N. (1993). "Distribution-Free" Estimates of Efficiency in the U.S. Banking Industry. *J. Productivity Analysis*, 4.

### CNLS / StoNED
- Hildreth, C. (1954). Point Estimates of Ordinates of Concave Functions. *J. American Statistical Association*, 49.
- Afriat, S. N. (1972). Efficiency Estimation of Production Functions. *International Economic Review*, 13.
- Kuosmanen, T. (2008). Representation Theorem for Convex Nonparametric Least Squares. *Econometrics Journal*, 11.
- Kuosmanen, T., Kortelainen, M. (2012). Stochastic Non-smooth Envelopment of Data: Semi-parametric Frontier Estimation. *J. Productivity Analysis*, 38.
- Kuosmanen, T., Johnson, A. L. (2010). DEA as Nonparametric Least-Squares Regression. *Operations Research*, 58.
- Park, B. U., Simar, L., Zelenyuk, V. (2015). Categorical Data in Local Maximum Likelihood: Theory and Applications to Productivity Analysis. *J. Productivity Analysis*, 43.

### ブートストラップ / 統計的推論
- Simar, L., Wilson, P. W. (1998). Sensitivity Analysis of Efficiency Scores: How to Bootstrap in Nonparametric Frontier Models. *Management Science*, 44.
- Simar, L., Wilson, P. W. (2000). Statistical Inference in Nonparametric Frontier Models: The State of the Art. *J. Productivity Analysis*, 13.
- Simar, L., Wilson, P. W. (2007). Estimation and Inference in Two-Stage, Semi-Parametric Models of Production Processes. *J. Econometrics*, 136.
- Daraio, C., Simar, L. (2005). Introducing Environmental Variables in Nonparametric Frontier Models. *J. Productivity Analysis*, 24.

### ML 系
- Esteve, M., Aparicio, J., Rabasa, A., Rodriguez-Sala, J. J. (2020). Efficiency Analysis Trees: A New Methodology for Estimating Production Frontiers Through Decision Trees. *Expert Systems with Applications*, 162.
- Esteve, M. et al. (2023). Random Forest for the Efficiency Analysis Trees (RFEAT). *European Journal of Operational Research*.
- Zheng, Y. et al. (2024). Robust Nonparametric Stochastic Frontier Analysis. *arXiv:2404.04301*.

### 動学・生産性指数
- Caves, D. W., Christensen, L. R., Diewert, W. E. (1982). The Economic Theory of Index Numbers and the Measurement of Input, Output, and Productivity. *Econometrica*, 50.
- Färe, R., Grosskopf, S., Lindgren, B., Roos, P. (1992). Productivity Changes in Swedish Pharmacies 1980-1989. *J. Productivity Analysis*, 3.
- Bjurek, H. (1996). The Malmquist Total Factor Productivity Index. *Scandinavian J. Economics*, 98.
- O'Donnell, C. J. (2014). Econometric Estimation of Distance Functions and Associated Measures of Productivity and Efficiency Change. *J. Productivity Analysis*, 41.
- Pastor, J. T., Lovell, C. A. K. (2005). A Global Malmquist Productivity Index. *Economics Letters*, 88.

### 望ましくない産出 / DDF
- Färe, R., Grosskopf, S., Lovell, C. A. K., Pasurka, C. (1989). Multilateral Productivity Comparisons When Some Outputs are Undesirable. *Review of Economics and Statistics*, 71.
- Chambers, R. G., Chung, Y., Färe, R. (1996). Benefit and Distance Functions. *J. Economic Theory*, 70.
- Chambers, R. G., Chung, Y., Färe, R. (1998). Profit, Directional Distance Functions, and Nerlovian Efficiency. *J. Optimization Theory and Applications*, 98.
- Tone, K. (2003). Dealing with Undesirable Outputs in DEA: A Slacks-Based Measure (SBM) Approach. *GRIPS Research Report Series*.

### 教科書
- Coelli, T. J., Rao, D. S. P., O'Donnell, C. J., Battese, G. E. (2005). *An Introduction to Efficiency and Productivity Analysis* (2nd ed.). Springer.
- Cooper, W. W., Seiford, L. M., Tone, K. (2007). *Data Envelopment Analysis: A Comprehensive Text* (2nd ed.). Springer.
- Bogetoft, P., Otto, L. (2011). *Benchmarking with DEA, SFA, and R*. Springer.
- Daraio, C., Simar, L. (2007). *Advanced Robust and Nonparametric Methods in Efficiency Analysis*. Springer.
- Kumbhakar, S. C., Lovell, C. A. K. (2003). *Stochastic Frontier Analysis*. Cambridge University Press.

---

## 改訂履歴

| 日付 | 改訂内容 |
|------|----------|
| 2026-05-07 | 初版作成。決定論・ノンパラ系手法を網羅し、SFA サーベイとの位置付けを明確化。実装計画は将来 Proposal を介して確定。 |
| 2026-05-07 | §11「動作検証用データセット」を追加。L1 (シミュレーション DGP)、L2 (公開データ)、L3 (外部) の三層構造、データ×手法対応マトリクス、検証プロトコル、同梱方針、ライセンス整理を含む。§12, §13 へ繰り下げ。 |
