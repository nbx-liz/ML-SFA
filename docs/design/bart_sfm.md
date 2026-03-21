# BART-SFM 詳細設計

**作成日:** 2026-03-22

---

## 1. 概要

ベイズ加法回帰木 (BART) で生産フロンティアをモデル化し、
SFA の合成誤差分解をベイズ推論で同時に行う手法。

フロンティアが回帰木のアンサンブルで表現されるため関数形の仮定が不要であり、
ベイズ的枠組みにより効率性推定値の事後分布（不確実性）が自然に得られる。

### 先行研究

| 文献 | 手法 | 特徴 |
|------|------|------|
| Wei, Sang, Coulibaly (2024) | monBART-SFM / softBART-SFM | 単調性制約付きBART + SFA |
| Chipman, George, McCulloch (2010) | BART | 基礎フレームワーク |
| Chipman et al. (2022) | mBART | 単調性制約付きBART |
| Linero, Yang (2018) | SoftBART | ソフト分割によるスムーズBART |

---

## 2. 数学的定式化

### 2.1 モデル

```
y_i = f(x_i) + v_i - u_i
```

ここで:

```
f(x_i) = Σ_{j=1}^{m} g_j(x_i; T_j, M_j)    (m本の回帰木の和)
v_i ~ N(0, σ_v²)                              (対称ノイズ)
u_i ~ N⁺(0, σ_u²)                             (半正規非効率性)
```

| 記号 | 定義 |
|------|------|
| `m` | 木の本数（通常 50-200） |
| `T_j` | 第 j 木の構造（分割ルール） |
| `M_j = {μ_1j, ..., μ_{b_j,j}}` | 第 j 木の葉パラメータ |
| `g_j(x_i; T_j, M_j)` | 第 j 木の予測値（x_i が到達する葉の μ 値） |

### 2.2 BART の正則化事前分布

各木が「弱い学習器」として機能するよう、3つの事前分布で正則化:

**1. 木構造の事前分布:**
深さ d のノードが非終端ノード（分割される）確率:
```
P(split at depth d) = α · (1 + d)^(-β)
```
デフォルト: `α = 0.95, β = 2` → 深い木を抑制

**2. 葉パラメータの事前分布:**
```
μ_ij ~ N(0, σ_μ²)
```
`σ_μ` は m 本の木の予測の和が y の範囲を覆うよう設定:
```
σ_μ = (y_max - y_min) / (2 · k · √m)    (k ≈ 2)
```

**3. 誤差分散の事前分布:**
```
σ_v² ~ InvGamma(ν/2, ν·λ/2)
```
データから事前にキャリブレーション。

### 2.3 非効率性分布の事前分布

```
u_i ~ N⁺(0, σ_u²)    (半正規: μ_u = 0)
σ_u² ~ InvGamma(a_u, b_u)
```

半正規は truncated normal の特殊ケース (`μ_u = 0`)。
`μ_u ≠ 0` の truncated normal への拡張も可能:
```
u_i ~ TN⁺(μ_u, σ_u²)    (一般化)
μ_u ~ N(0, τ²)
```

---

## 3. 推論アルゴリズム

### 3.1 データ拡張付きギブスサンプラー

潜在変数 `u_i` を明示的にサンプリングするデータ拡張戦略を用いる。

```
MCMC iteration t = 1, 2, ..., T:

  Step 1: 各木 j = 1, ..., m について (バックフィッティング):
    (a) 部分残差を計算:
        R_j = y - Σ_{k≠j} g_k(x; T_k, M_k) + u
              ↑ 他の木の予測の和を引く    ↑ 非効率性を足し戻す

    (b) 木構造 T_j を更新:
        Metropolis-Hastings で grow/prune/change 提案 → 受容/棄却

    (c) 葉パラメータ M_j をサンプリング:
        μ_ij | T_j, R_j, σ_v² ~ N(μ̃_ij, σ̃²_j)
        (制約なしBART: 正規分布から直接。monBART: 切断正規分布)

  Step 2: 潜在非効率性 u_i をサンプリング:
    u_i | rest ~ TN⁺(μ*_i, σ*²)    (正の切断正規)

    μ*_i = -(y_i - f(x_i)) · σ_u² / σ²
    σ*² = σ_u² · σ_v² / σ²
    σ² = σ_u² + σ_v²

  Step 3: 分散パラメータをサンプリング:
    σ_v² | rest ~ InvGamma(...)    (正規-逆ガンマ共役)
    σ_u² | rest ~ InvGamma(...)    (半正規の u_i からの十分統計量)
```

### 3.2 木構造の更新（Metropolis-Hastings）

各ステップで以下の提案の一つをランダムに選択:

| 提案 | 操作 | 効果 |
|------|------|------|
| **Grow** | 葉ノードを内部ノードに変換（分割を追加） | 木を深くする |
| **Prune** | 内部ノードを葉に変換（分割を除去） | 木を浅くする |
| **Change** | 分割変数または分割点を変更 | 木構造を修正 |

受容確率は Metropolis-Hastings 比:
```
α = min(1, [p(R_j | T_j*, M_j*, σ²) · p(T_j*)] / [p(R_j | T_j, M_j, σ²) · p(T_j)])
    × [q(T_j | T_j*) / q(T_j* | T_j)]
```

周辺尤度 `p(R_j | T_j, σ²)` は葉パラメータ M_j を積分消去して閉じた形で計算可能
（正規-正規共役）。

### 3.3 効率性の事後分布

各 MCMC イテレーション t で:
```
u_i^(t) → TE_i^(t) = exp(-u_i^(t))
```

T 回のイテレーション後:
```
E[TE_i | data] ≈ (1/T) Σ_t TE_i^(t)           (事後平均)
CI(TE_i) = [quantile(TE_i^(t), 0.025), quantile(TE_i^(t), 0.975)]  (95%信用区間)
```

これがベイズアプローチの大きな利点: **効率性の不確実性が自然に定量化される。**

---

## 4. 2つのバリアント

### 4.1 monBART-SFM（単調性制約付き）

Chipman et al. (2022) の mBART に基づく。

**制約**: 指定された入力変数について `f(x)` が単調非減少:
```
x_j ≤ x_j' ⇒ f(x) ≤ f(x')  (j ∈ 単調変数の集合 S)
```

**実装**:
- 各木の葉パラメータに順序制約: 単調変数の方向で葉値が非減少
- 制約集合 `C = {(T, M) : g(x; T, M) is monotone}` を事前分布に組み込む
- 葉パラメータのサンプリングが **切断正規分布** になる（共役性が崩れる）

**利点**: 経済学的に妥当な単調性を保証
**欠点**: MCMC の効率低下、凹性は保証しない

### 4.2 softBART-SFM（スムーズ型）

Linero & Yang (2018) の SoftBART に基づく。

**ハード分割の代わりにソフト分割:**
```
ψ(x; c, τ) = 1 / (1 + exp(-(x - c)/τ))
```
- `c`: 分割点
- `τ`: バンド幅（τ → 0 でハード分割に収束）

各観測は確率的に全ての葉に割り当てられ、予測は全パスの加重平均。

**利点**: 滑らかな予測、異なる滑らかさレベルに自動適応
**欠点**: 明示的な単調性保証なし（データと事前分布に依存して近似）、
          PyMC-BART は標準 (ハード) BART であり SoftBART 未実装

### 4.3 バリアント選択の指針

| 基準 | monBART-SFM | softBART-SFM |
|------|:-----------:|:------------:|
| 単調性保証 | **保証** | 近似的 |
| 滑らかさ | 区分定数 | **連続的** |
| 実装の容易さ | 中（切断正規サンプリング） | 難（SoftBART 自体の実装が必要） |
| PyMC-BART 互換 | 修正が必要（単調性制約追加） | 未対応（SoftBART 未実装） |
| 推奨状況 | 解釈性・単調性が重要な場合 | 滑らかなフロンティアが想定される場合 |

**→ 実装上の実現可能性から、まず monBART-SFM 相当の実装を推奨。**
**→ ただし PyMC-BART に単調性制約がないため、データ拡張アプローチで回避（後述）。**

---

## 5. PyMC 実装戦略

### 5.1 アプローチ: データ拡張 + PyMC-BART

PyMC-BART に単調性制約がないため、以下の戦略を取る:

1. `pmb.BART` でフロンティアをモデル化（制約なし — softBART-SFM 的）
2. 潜在変数 `u_i` を明示的にモデルに組み込み
3. PyMC の NUTS サンプラーが `σ_v, σ_u, u_i` を、PGBART が木を同時サンプリング

```python
import pymc as pm
import pymc_bart as pmb
import numpy as np

with pm.Model() as bart_sfa:
    # --- 分散パラメータの事前分布 ---
    sigma_v = pm.HalfNormal('sigma_v', sigma=1.0)
    sigma_u = pm.HalfNormal('sigma_u', sigma=1.0)

    # --- フロンティア (BART) ---
    # X: (n, p) 入力, Y: (n,) 産出
    mu = pmb.BART('mu', X=X_train, Y=Y_train, m=50)

    # --- 潜在非効率性 ---
    u = pm.HalfNormal('u', sigma=sigma_u, shape=n)

    # --- 観測モデル ---
    y_obs = pm.Normal('y_obs', mu=mu - u, sigma=sigma_v, observed=Y_train)
```

### 5.2 代替: 周辺尤度アプローチ

潜在変数 u を積分消去し、合成誤差の閉じた密度を使用:

```python
import pytensor.tensor as pt

with pm.Model() as bart_sfa_marginal:
    sigma_v = pm.HalfNormal('sigma_v', sigma=1.0)
    sigma_u = pm.HalfNormal('sigma_u', sigma=1.0)

    mu = pmb.BART('mu', X=X_train, Y=Y_train, m=50)

    sigma = pt.sqrt(sigma_v**2 + sigma_u**2)
    lam = sigma_u / sigma_v
    eps = Y_train - mu

    # SFA 合成誤差の対数密度
    log_lik = (
        pt.log(2) - pt.log(sigma)
        + pm.logp(pm.Normal.dist(), eps / sigma)
        + pm.logcdf(pm.Normal.dist(), -eps * lam / sigma)
    )
    pm.Potential('sfa_likelihood', log_lik.sum())
```

周辺尤度の利点: 潜在変数が n 個減り、サンプリング効率向上。
欠点: `u_i` の事後分布が直接得られないため、JLMS 公式で事後的に計算が必要。

### 5.3 効率性の事後推定

**データ拡張アプローチ (5.1):**
```python
trace = pm.sample(2000, tune=1000)
u_posterior = trace.posterior['u']                    # (chains, draws, n)
te_posterior = np.exp(-u_posterior)                   # TE の事後分布
te_mean = te_posterior.mean(dim=('chain', 'draw'))   # 事後平均
```

**周辺尤度アプローチ (5.2):**
各事後サンプルで JLMS:
```python
for draw in posterior_draws:
    sigma_v_s, sigma_u_s = draw['sigma_v'], draw['sigma_u']
    mu_s = draw['mu']
    eps_s = Y - mu_s
    sigma_sq = sigma_v_s**2 + sigma_u_s**2
    mu_star = -eps_s * sigma_u_s**2 / sigma_sq
    sigma_star = sigma_v_s * sigma_u_s / np.sqrt(sigma_sq)
    E_u_s = jlms(mu_star, sigma_star)
    TE_s = np.exp(-E_u_s)
```

### 5.4 計算コストの見積もり

| 項目 | データ拡張 | 周辺尤度 |
|------|:----------:|:--------:|
| パラメータ数 | n + 2 + 木構造 | 2 + 木構造 |
| サンプリング速度 | 遅（u_i が n 個） | 中 |
| TE の直接推定 | あり | 事後計算が必要 |
| 推奨 N | N < 2,000 | N < 5,000 |

**N = 500, m = 50 の場合の概算:**
- データ拡張: 約 30-60 分 (2000 draws + 1000 tuning)
- 周辺尤度: 約 15-30 分

---

## 6. BART-SFM の診断

### 6.1 収束診断

| 診断 | 対象 | 方法 |
|------|------|------|
| R̂ (Gelman-Rubin) | σ_v, σ_u, u_i | 複数チェーンで R̂ < 1.01 |
| ESS (有効サンプルサイズ) | σ_v, σ_u | ESS > 400 |
| トレースプロット | σ_v, σ_u, ln L | 定常性の目視確認 |
| 木の深さ・葉数 | T_j | イテレーション間での安定性 |

### 6.2 モデル診断

| 診断 | 内容 |
|------|------|
| 部分依存プロット (PDP) | `pmb.plot_pdp(...)` で各投入と産出の関係を可視化 |
| 変数重要度 | `pmb.plot_variable_importance(...)` |
| 残差分析 | 合成誤差 ε の分布が歪正規に従うか |
| 効率性の事後分布 | TE_i の信用区間幅で推定の不確実性を評価 |

---

## 7. 同時推定NNとの比較

| 観点 | 同時推定NN | BART-SFM |
|------|:----------:|:--------:|
| **フロンティア表現** | 連続的な NN | 区分定数（木のアンサンブル） |
| **推論方法** | 勾配ベース MLE | MCMC (ギブスサンプリング) |
| **不確実性定量化** | なし（点推定） | **あり（事後分布）** |
| **単調性制約** | 重み制約 or ペナルティ | monBART / ソフト近似 |
| **解釈性** | PDP, SHAP で事後解釈 | **変数重要度・PDP が自然** |
| **少量データ** | 不安定 (N > 500 推奨) | **安定 (N > 100 で実用的)** |
| **大規模データ** | **スケーラブル** | 困難 (N > 10,000 で遅い) |
| **計算時間** | 数分〜数十分 | 数十分〜数時間 |
| **局所最適の問題** | あり（多重初期化で緩和） | なし（MCMC は大域探索） |
| **実装の複雑さ** | 中（PyTorch + カスタム損失） | 中〜高（PyMC + PyMC-BART） |

**相補的な関係**: NN は大規模・高次元データに、BART は少量データ・不確実性が重要な場合に適する。

---

## 8. 実装上の注意点

### 8.1 PyMC-BART の制限と対策

| 制限 | 対策 |
|------|------|
| 単調性制約なし | softBART 的アプローチ（制約なし）で開始。結果の単調性を事後確認 |
| SoftBART 未実装 | 標準 BART (ハード分割) を使用。m を増やして滑らかさを改善 |
| 計算速度 | m=50 から開始。PyMC の nutpie バックエンド (Rust) で高速化 |
| メモリ | データ拡張で n 個の潜在変数 → N > 5,000 ではメモリ注意 |

### 8.2 ハイパーパラメータ

| パラメータ | 推奨値 | 調整指針 |
|-----------|--------|---------|
| 木の本数 m | 50 | 滑らかさ不足なら 100-200 |
| α (分割確率) | 0.95 | デフォルト推奨 |
| β (深さペナルティ) | 2 | デフォルト推奨 |
| MCMC draws | 2,000 | R̂ が収束するまで増加 |
| Tuning steps | 1,000 | draws の半分以上 |
| チェーン数 | 4 | 収束診断に必要 |

### 8.3 検証すべき事項

- [ ] シミュレーション DGP で真の TE との RMSE を計測
- [ ] パラメトリック SFA との比較（正しい関数形 DGP / 誤った関数形 DGP）
- [ ] 同時推定 NN-SFA との比較（同一 DGP）
- [ ] TE の信用区間のカバー率（真の TE が CI に含まれる割合）
- [ ] 変数重要度の正確性（真のモデルの既知構造と比較）
- [ ] サンプルサイズ (N = 100, 200, 500, 1000) での挙動
- [ ] 木の本数 m の感度分析
- [ ] データ拡張 vs 周辺尤度アプローチの比較

---

## 9. 参考文献

- Wei, Z., Sang, H., Coulibaly, N. (2024). Nonparametric Machine Learning for Stochastic Frontier Analysis: A Bayesian Additive Regression Tree Approach. *Econometrics and Statistics*. DOI: 10.1016/j.ecosta.2024.06.002
- Chipman, H.A., George, E.I., McCulloch, R.E. (2010). BART: Bayesian Additive Regression Trees. *Annals of Applied Statistics*, 4(1), 266-298.
- Chipman, H.A., George, E.I., McCulloch, R.E., Shively, T.S. (2022). mBART: Multidimensional Monotone BART. *Bayesian Analysis*, 17(2), 515-544.
- Linero, A.R., Yang, Y. (2018). Bayesian Regression Tree Ensembles that Adapt to Smoothness and Sparsity. *JRSS-B*, 80(5), 1087-1110.
- Quiroga, M. et al. (2023). Bayesian Additive Regression Trees for Probabilistic Programming. *arXiv:2206.03619*.
- He, J., Yalov, S. (2019). XBART: Accelerated Bayesian Additive Regression Trees. *AISTATS 2019*.
- Jondrow, J. et al. (1982). On the Estimation of Technical Inefficiency. *Journal of Econometrics*, 19(2-3), 233-238.
