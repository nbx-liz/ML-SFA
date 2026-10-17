# ML-SFA 用語集 (Glossary)

**目的:** 本プロジェクトで使用する用語を統一し、ドキュメント・docstring・テスト間で
意味のずれを防ぐ。サーベイ ([`research/ml_sfa_survey.md`](research/ml_sfa_survey.md),
[`research/deterministic_frontier_survey.md`](research/deterministic_frontier_survey.md))
と本ファイルは双方向に参照される。

各用語の英訳・略称・出典・本ライブラリでの実装名/属性名を併記する。

---

## 1. 効率性概念

### 1.1 Technical Efficiency (TE) — 技術的効率性

| 項目 | 内容 |
|------|------|
| 定義 | 与えられた投入量で最大限の産出を得ているか (または与えられた産出を最小投入で達成しているか) |
| 値域 | (0, 1] (input-oriented) または [1, ∞) (output-oriented)。1 が完全効率 |
| 出典 | Farrell (1957), Shephard (1953) |
| 実装 | `BaseFrontierEstimator.efficiency()` の戻り値 |

### 1.2 Allocative Efficiency (AE) — 配分効率性

| 項目 | 内容 |
|------|------|
| 定義 | 投入要素の比率が要素価格に照らして最適か |
| 必要情報 | 投入価格ベクトル `w` |
| 関係 | Cost Efficiency = TE × AE |
| 実装 | 価格情報が必要なため Phase 後半で個別 Proposal を介して実装予定 |

### 1.3 Scale Efficiency (SE) — 規模効率性

| 項目 | 内容 |
|------|------|
| 定義 | 最適規模に対する乖離。SE = TE_CRS / TE_VRS |
| 値域 | (0, 1] |
| 実装 | `ml_sfa.frontier.envelopment.scale_efficiency()` ヘルパで提供予定 |

### 1.4 Cost / Revenue / Profit Efficiency

| 名称 | 定義 |
|------|------|
| Cost Efficiency | TE × AE (入力面) |
| Revenue Efficiency | 産出側 TE × 産出 AE |
| Profit Efficiency | DDF ベースで計測 (Chambers, Chung, Färe 1998) |

---

## 2. 距離関数 (Distance Functions)

### 2.1 Farrell efficiency

| 項目 | 内容 |
|------|------|
| 定義 (input) | `θ_F(x, y) = min { θ > 0 \| (θ x, y) ∈ T }` |
| 定義 (output) | `φ_F(x, y) = max { φ > 0 \| (x, φ y) ∈ T }` |
| 関係 | input efficiency = `θ_F`、output efficiency = `1 / φ_F` |
| 出典 | Farrell (1957) |

### 2.2 Shephard input/output distance function

| 項目 | 内容 |
|------|------|
| Input distance | `D_I(x, y) = sup { λ > 0 \| (x/λ, y) ∈ T }` |
| Output distance | `D_O(x, y) = inf { φ > 0 \| (x, y/φ) ∈ T }` |
| 関係 | `θ_F = 1 / D_I`、`φ_F = 1 / D_O` |
| 出典 | Shephard (1953, 1970) |

### 2.3 Directional Distance Function (DDF)

| 項目 | 内容 |
|------|------|
| 定義 | `D_g(x, y) = sup { β \| (x - β g_x, y + β g_y) ∈ T }` |
| 方向ベクトル | `g = (-g_x, g_y)`、または望ましくない産出 `b` を含めて `g = (-g_x, g_y, -g_b)` |
| 効率的 | `β* = 0` |
| 出典 | Chambers, Chung, Färe (1996, 1998) |
| 実装 | `ml_sfa.frontier.ddf.DDF` |

---

## 3. モデル仕様用語

### 3.1 Orientation (指向)

| 値 | 意味 | スコア値域 |
|----|------|-----------|
| `"input"` | 産出を維持して投入を最小化 | (0, 1] |
| `"output"` | 投入を維持して産出を最大化 | [1, ∞) |
| `"hyperbolic"` | 投入縮小 + 産出拡大の同時実行 (Färe et al. 1985) | (0, 1] |
| `"non-oriented"` | スラックベース、無指向 (SBM 等) | (0, 1] |

### 3.2 Returns to Scale (規模に関する仮定)

| 値 | 制約 (DEA 双対) | 経済学的解釈 |
|----|----------------|---------------|
| `"crs"` | `Σ λ_j` 制約なし | 全 DMU が最適規模で操業 (Constant Returns to Scale) |
| `"vrs"` | `Σ λ_j = 1` | 規模に関する仮定なし (Variable Returns to Scale) |
| `"nirs"` | `Σ λ_j ≤ 1` | 非増加収穫 (Non-Increasing Returns to Scale) |
| `"ndrs"` | `Σ λ_j ≥ 1` | 非減少収穫 (Non-Decreasing Returns to Scale) |

### 3.3 Disposability (処分可能性)

| 種別 | 内容 |
|------|------|
| Strong / Free disposability | `(x, y) ∈ T`, `x' ≥ x`, `y' ≤ y` ⇒ `(x', y') ∈ T` |
| Weak disposability | 望ましくない産出 `b` を扱う際、`y` と `b` の同時拡張・縮減を要求 (Färe et al. 1989) |
| 用途 | 環境効率分析、CO2 排出を含む生産技術 |

---

## 4. 推定構造用語

### 4.1 DMU (Decision Making Unit)

| 項目 | 内容 |
|------|------|
| 意味 | 意思決定単位 (企業、病院、銀行支店、学校など評価対象) |
| 数学的表現 | `(x_i, y_i) ∈ R_+^p × R_+^q`、`i = 1, ..., n` |
| 実装属性 | `n_dmus_`, `X_`, `Y_` |

### 4.2 Peer set (ピア集合)

| 項目 | 内容 |
|------|------|
| 定義 | DMU `i` の効率性計算で `λ_j > 0` となる `j` の集合 |
| 解釈 | DMU `i` のベンチマーク対象 |
| 実装属性 | `peers_: list[NDArray[int]]` (ndarray 入力) または `pd.Series[list[str]]` (DataFrame 入力) |

### 4.3 Slack (スラック)

| 項目 | 内容 |
|------|------|
| 入力スラック `s_i^-` | 投入 `x_i0` を効率点まで縮小したあと、さらに削減可能な余剰 |
| 産出スラック `s_r^+` | 産出 `y_r0` を効率点まで拡大したあと、さらに増加可能な余地 |
| 実装属性 | `slacks_input_: (n, p)`, `slacks_output_: (n, q)` |

### 4.4 Super-efficiency (スーパー効率性)

| 項目 | 内容 |
|------|------|
| 定義 | 対象 DMU を参照集合から除外して計算した効率性 |
| 値域 | (0, ∞) — 1 を超えうる |
| 出典 | Andersen, Petersen (1993) for radial、Tone (2002) for SBM |
| 実装 | `SuperCCR`, `SuperBCC`, `SuperSBM` (子クラス) |

### 4.5 Convex hull / Free Disposal Hull

| 項目 | 内容 |
|------|------|
| Convex hull (DEA) | 観測点の凸結合 + free disposability で形成される技術集合 |
| Free Disposal Hull (FDH) | free disposability のみ、凸性を仮定しない技術集合 (Deprins, Simar, Tulkens 1984) |
| 関係 | `T_FDH ⊆ T_DEA-VRS ⊆ T_DEA-CRS` |

### 4.6 Frontier shift / Catching-up (Malmquist 分解)

| 項目 | 内容 |
|------|------|
| Catching-up (efficiency change) | 自 DMU が時間とともにフロンティアに近づく効果 |
| Frontier shift (technical change) | フロンティア自体の移動 (技術進歩 / 退歩) |
| 関係 | MPI = catching-up × frontier shift |
| 出典 | Färe et al. (1992) |
| 実装 | `MalmquistIndex.efficiency_change_`, `MalmquistIndex.frontier_shift_` |

---

## 5. 推定誤差項 (SFA 系で主に使用)

### 5.1 Composed error (合成誤差)

| 項目 | 内容 |
|------|------|
| 定義 | `ε = v - u`、`v` はノイズ、`u ≥ 0` は非効率性 |
| 出典 | Aigner, Lovell, Schmidt (1977), Meeusen, van den Broeck (1977) |
| 関連 | Jondrow et al. (1982) の `E[u \| ε]` |

### 5.2 Inefficiency distribution (非効率性分布)

| 名称 | 表記 | 用途 |
|------|------|------|
| Half-normal | `u ~ \|N(0, σ_u²)\|` | 既定 |
| Truncated normal | `u ~ N⁺(μ, σ_u²)` | モード ≠ 0 を許容 |
| Exponential | `u ~ Exp(λ)` | 単純な代替 |
| Gamma | `u ~ Gamma(P, λ)` | 最も柔軟 |

---

## 6. ハイパーパラメータの 5 分類タグ

`HISTORY.md P-0001 D-0001-10` で確定。全 docstring に付与。

| タグ | 意味 | 既定値の方針 | チューニング |
|------|------|-------------|-------------|
| `[STRUCTURAL]` | モデル定義の一部 (orientation, returns, weights, direction) | 慎重に決め、デフォルトを明示 | 対象外 |
| `[INFERENCE]` | 統計的推論 (n_bootstrap, bandwidth, confidence_level) | 理論的既定値 (Silverman, B=2000 等) | 通常変更不要 |
| `[ROBUSTNESS]` | 外れ値耐性 (m, alpha, quantile_loss_tau) | `auto` モード (Daouia heuristic 等) | ユーザの分析意図 |
| `[ML-HP]` | ML 系 (max_depth, n_estimators, learning_rate, lambda) | sklearn 標準値 | CV (frontier-aware scorer) |
| `[COMPUTATIONAL]` | 数値計算 (solver, tol, max_iter) | 「まず動く」値、`solver_options=` dict | エラー時のみ |

### docstring での書式例

```python
class CCR:
    """DEA-CCR (Charnes-Cooper-Rhodes 1978) model.

    See deterministic_frontier_survey.md §3.2.

    Parameters
    ----------
    orientation : {"input", "output"}, default="input"
        [STRUCTURAL] Direction of efficiency measurement. Cannot be tuned.
    solver : str, default="highs"
        [COMPUTATIONAL] LP solver. Use ``solver_options=`` for fine-tuning.
    """
```

---

## 7. 効率性スコア値域 一覧

| クラス | Family | 値域 | 効率的 (= ?) | 備考 |
|--------|--------|:---:|:---:|------|
| `CCR` (input) | Envelopment | (0, 1] | 1 | radial |
| `CCR` (output) | Envelopment | [1, ∞) | 1 | radial |
| `BCC` (input) | Envelopment | (0, 1] | 1 | radial + VRS |
| `BCC` (output) | Envelopment | [1, ∞) | 1 | radial + VRS |
| `Additive` | Envelopment | [0, ∞) | 0 | スラック和、非ラジアル |
| `SBM` | Envelopment | (0, 1] | 1 | スラックベース |
| `SuperCCR` / `SuperBCC` | Envelopment | (0, ∞) | 1 を超える | self 除外 |
| `SuperSBM` | Envelopment | (0, ∞) | 1 を超える | Tone 2002 |
| `FDH` | Envelopment | (0, 1] | 1 | 非凸 |
| `OrderM` | Robust | (0, ∞) | ≤ 1 で効率的 | m → ∞ で FDH |
| `OrderAlpha` | Robust | (0, ∞) | ≤ 1 で効率的 | α 分位 |
| `COLS` / `MOLS` | Functional | (0, 1] | 1 | TE = exp(-u) |
| `TFA` / `DFA` | Functional | (0, 1] | 1 | パネル平均ベース |
| `CNLS` | Functional | (0, 1] | 1 | ノンパラ凹単調 |
| `StoNED` | Functional | (0, 1] | 1 | CNLS + SFA |
| `EAT` / `RFEAT` | Functional | (0, 1] | 1 | 階段状フロンティア |
| `DDF` | Directional | [0, ∞) | 0 | β* (距離値) |

**読み方:**
- `BootstrapDEA` (Wrapper) は内部 base estimator の値域を継承
- `CrossEfficiency` (Wrapper) は (0, 1] 固定 (列平均化のため)

---

## 8. 略語表

| 略語 | 正式名 | 関連節 |
|------|--------|--------|
| AE | Allocative Efficiency | §1.2 |
| BCC | Banker-Charnes-Cooper (1984) | sub-DEA-VRS |
| CCR | Charnes-Cooper-Rhodes (1978) | sub-DEA-CRS |
| CNLS | Convex Nonparametric Least Squares | sub-Functional |
| COLS | Corrected Ordinary Least Squares | sub-Functional |
| CRS | Constant Returns to Scale | §3.2 |
| DDF | Directional Distance Function | §2.3 |
| DEA | Data Envelopment Analysis | family |
| DFA | Distribution-Free Approach | sub-Functional |
| DGP | Data Generating Process | テスト用語 |
| DMU | Decision Making Unit | §4.1 |
| DRS | Decreasing Returns to Scale | §3.2 |
| EAT | Efficiency Analysis Trees | sub-Functional |
| FDH | Free Disposal Hull | family |
| IRS | Increasing Returns to Scale | §3.2 |
| LOOCV | Leave-One-Out Cross-Validation | tuning |
| LP | Linear Programming | computational |
| MOLS | Modified Ordinary Least Squares | sub-Functional |
| MPI | Malmquist Productivity Index | §4.6 |
| NDRS | Non-Decreasing Returns to Scale | §3.2 |
| NIRS | Non-Increasing Returns to Scale | §3.2 |
| NN-SFA | Neural Network SFA | (ml_sfa_survey) |
| QP | Quadratic Programming | computational |
| RFEAT | Random Forest EAT | sub-Functional |
| SBM | Slacks-Based Measure | sub-DEA |
| SE | Scale Efficiency | §1.3 |
| SFA | Stochastic Frontier Analysis | (ml_sfa_survey) |
| StoNED | Stochastic Nonsmooth Envelopment of Data | sub-Functional |
| TE | Technical Efficiency | §1.1 |
| TFA | Thick Frontier Approach | sub-Functional |
| VRS | Variable Returns to Scale | §3.2 |

---

## 9. 実装命名規約 (本ライブラリ固有)

### 9.1 fitted attributes (sklearn 慣習: 末尾 `_`)

| 属性 | 型 | 全クラス必須? | 説明 |
|------|----|:---:|------|
| `X_` | `NDArray (n, p)` | ◎ | 訓練データ投入 |
| `Y_` | `NDArray (n, q)` | ◎ | 訓練データ産出 |
| `n_dmus_` | `int` | ◎ | DMU 数 |
| `n_inputs_` | `int` | ◎ | 投入変数数 |
| `n_outputs_` | `int` | ◎ | 産出変数数 |
| `efficiency_` | `NDArray (n,)` | ◎ | 訓練データの効率値 |
| `efficiency_range_` | `tuple[float, float]` | ◎ | 効率値の値域 |
| `lambdas_` | `NDArray (n, n)` | Envelopment のみ | 凸結合重み |
| `slacks_input_` | `NDArray (n, p)` | Envelopment のみ | 入力スラック |
| `slacks_output_` | `NDArray (n, q)` | Envelopment のみ | 産出スラック |
| `peers_` | `list[NDArray[int]]` | Envelopment のみ | ピアインデックス |
| `efficiency_distribution_` | `NDArray (B, n)` | Robust / Wrapper のみ | ブートストラップ分布 |
| `coef_` / `intercept_` | various | Functional のみ | 回帰係数 |
| `frontier_values_` | `NDArray (n,)` | Functional のみ | f̂(X_) |
| `sigma_u_` / `sigma_v_` | `float` | StoNED のみ | 分散成分 |

### 9.2 メソッド命名規約

| メソッド | 全 family? | 備考 |
|---------|:---:|------|
| `fit(X, Y, *, inputs=, outputs=)` | ◎ | DataFrame 統合 |
| `efficiency(X=None, Y=None)` | ◎ | None で訓練データ |
| `project(X, Y)` | ◎ | フロンティア射影 |
| `summary()` | ◎ | `FrontierSummary` を返す |
| `score(X, Y)` | ◎ | sklearn 互換 |
| `predict(X)` | Functional のみ | 単一産出のみ |
| `slacks(X=None, Y=None)` | Envelopment のみ | (s^-, s^+) を返す |
| `peers(X=None, Y=None)` | Envelopment のみ | ピアインデックス |
| `confidence_interval(level=0.95)` | Robust / Bootstrap Wrapper | (lower, upper) |
| `bootstrap_distribution()` | Robust / Bootstrap Wrapper | 分布 |
| `frontier(X)` | Functional のみ | predict のエイリアス (将来) |

### 9.3 入出力対称性 (D-0001-03)

| 入力 | 出力 |
|------|------|
| ndarray | ndarray |
| DataFrame + `inputs=`/`outputs=` | pd.Series / pd.DataFrame (DMU index 保持) |

---

## 10. テスト用語 (`tests/data/` 配下)

| 用語 | 意味 |
|------|------|
| L1 | シミュレーション DGP データ (CI 必須) |
| L2 | 公開データ (再配布可能、リポジトリ同梱) |
| L3 | 外部ダウンロード (オプション、CI 非依存) |
| Reproduction test | 公表値との数値一致テスト |
| Property test | 不変条件 (単調性、scale invariance 等) のテスト |
| Cross-software test | 他ソフトウェア (pyStoNED, R Benchmarking 等) との数値整合テスト |

詳細は [`research/deterministic_frontier_survey.md` §11](research/deterministic_frontier_survey.md#11-動作検証用データセット) を参照。

---

## 改訂履歴

| 日付 | 改訂内容 |
|------|----------|
| 2026-05-07 | 初版作成。HISTORY.md P-0001 と同期。サーベイ 2 本と双方向リンク。ハイパーパラメータ 5 分類、命名規約、効率値域一覧を含む。 |
