# ML-SFA 設計図 (Blueprint)

## 1. プロジェクト概要

### 目的

ML手法を用いた確率的フロンティア分析 (Stochastic Frontier Analysis) のPythonライブラリを構築する。
従来のパラメトリックSFAの制約（関数形依存・分布仮定への感度）をML技術で緩和し、
scikit-learn互換APIで提供する。

### 背景: 従来のSFA

SFAは Aigner, Lovell, Schmidt (1977) および Meeusen, van den Broeck (1977) が提案した手法で、
生産フロンティアからの乖離を **ノイズ (v)** と **非効率性 (u)** に分解する:

```
y_i = f(x_i; β) + v_i - u_i
```

- `f(x_i; β)`: 決定論的フロンティア（Cobb-Douglas, translog等）
- `v_i ~ N(0, σ_v²)`: 対称ノイズ（測定誤差、確率的ショック）
- `u_i ≥ 0`: 片側非効率性（half-normal, truncated normal, exponential等）
- 技術的効率性: `TE_i = exp(-u_i)`

**従来手法の限界:**
1. フロンティア関数形への依存（Cobb-Douglas vs translog で結果が変わる）
2. 非効率性の分布仮定への感度
3. 複雑な非線形投入・産出関係を捉えられない
4. v と u の分離が分布仮定に完全依存（脆弱な識別）

### 既存実装の空白

| パッケージ | 特徴 | 欠落 |
|-----------|------|------|
| pySFA | 基本的MLE (half-normal のみ) | ML拡張なし、パネルなし |
| FronPy | 6分布サポート | 線形フロンティアのみ |
| SFMA | B-spline + shape制約 | メタ分析特化 |
| pyStoNED | CNLS + SFA分解 | ML手法なし |

**→ ML-SFA が埋めるギャップ:**
- scikit-learn互換APIのSFAライブラリ
- NN / GP / BART 等によるノンパラメトリックフロンティア
- 経済学的shape制約（単調性・凹性）
- パネルデータ対応
- 不確実性定量化

---

## 2. アーキテクチャ

### 2.1 全体構成

```
ml_sfa/
├── models/
│   ├── base.py            # 基底クラス (BaseSFAEstimator)
│   ├── parametric.py      # 伝統的パラメトリックSFA
│   ├── nn_frontier.py     # ニューラルネットワークSFA
│   ├── gp_frontier.py     # ガウス過程SFA
│   └── kernel_frontier.py # カーネルSFA
├── data/
│   ├── loader.py          # データ読み込み・検証
│   ├── preprocessor.py    # 前処理パイプライン
│   └── simulator.py       # シミュレーションデータ生成
├── evaluation/
│   ├── metrics.py         # 効率性・モデル評価指標
│   └── comparison.py      # モデル比較フレームワーク
└── utils/
    ├── distributions.py   # 非効率性分布 (half-normal, truncated normal, etc.)
    └── constraints.py     # Shape制約 (単調性, 凹性)
```

### 2.2 クラス階層

```
sklearn.base.BaseEstimator
└── BaseSFAEstimator (ABC)
    ├── fit(X, y) → self
    ├── predict(X) → y_hat (フロンティア予測)
    ├── efficiency(X, y) → TE (技術的効率性)
    ├── get_inefficiency(X, y) → u_hat
    ├── get_noise(X, y) → v_hat
    ├── log_likelihood() → float
    ├── summary() → SFASummary
    │
    ├── ParametricSFA
    │   ├── frontier_type: "cobb-douglas" | "translog"
    │   └── inefficiency_dist: "half-normal" | "truncated-normal" | "exponential"
    │
    ├── NNFrontierSFA
    │   ├── フロンティア: MLP / RBF ネットワーク
    │   └── 誤差分解: MLE (2段階) or 同時推定
    │
    ├── GPFrontierSFA
    │   ├── フロンティア: ガウス過程回帰 (GPyTorch)
    │   └── 単調性: virtual derivative observations
    │
    └── KernelSFA
        ├── フロンティア: Nadaraya-Watson / 局所多項式
        └── 誤差分解: 局所MLE or 2段階
```

### 2.3 データフロー

```
入力データ (X, y)
    │
    ▼
[前処理] → 標準化, 対数変換, 欠損値処理
    │
    ▼
[フロンティア推定] → f̂(X) をML手法で推定
    │                  (shape制約: 単調増加, 凹性)
    ▼
[残差計算] → ε̂_i = y_i - f̂(x_i)
    │
    ▼
[誤差分解] → ε̂ = v̂ - û
    │          (分布仮定 or ノンパラメトリック)
    ▼
[効率性推定] → TE_i = exp(-û_i)
    │            E[u_i | ε_i] (Jondrow et al. 1982)
    ▼
出力: TE, û, v̂, f̂(X), モデル診断
```

---

## 3. モデル設計

### Phase 1: 基盤 (ParametricSFA)

従来のパラメトリックSFAをscikit-learn APIで実装する。
これが全MLモデルのベンチマーク基準となる。

**仕様:**
- フロンティア関数: Cobb-Douglas, Translog
- 非効率性分布: Half-normal, Truncated normal, Exponential
- 推定: MLE (scipy.optimize + 解析的勾配)
- 効率性推定: Jondrow et al. (1982) 条件付き期待値
- パネルデータ: 固定効果, ランダム効果 (将来)

**核心となる対数尤度 (half-normal ケース):**
```
ln L = const - N ln σ + Σ[ ln Φ(-ε_i λ/σ) - ε_i²/(2σ²) ]

σ² = σ_v² + σ_u²,  λ = σ_u / σ_v
```

### Phase 2: ニューラルネットワークSFA (NNFrontierSFA)

**アプローチ A: 2段階NN-SFA** (Kutlu 2024 に基づく)
1. Stage 1: MLPでフロンティア `f̂(x)` を推定（単調性制約付き）
2. Stage 2: 残差に対してパラメトリック誤差分解

**アプローチ B: 同時推定** (Tsionas et al. 2023 に基づく)
- フロンティアNNと誤差分解を同時最適化
- カスタム損失関数にSFA対数尤度を埋め込む
- PyTorch autograd で勾配計算

**アプローチ C: 構造化NN** (xNN-SF, Zhao 2024 に基づく)
- SFA分解構造をネットワークアーキテクチャに反映
- フロンティア・非効率性・ノイズの3サブネット
- 単調性制約をアーキテクチャレベルで強制

**単調性制約の実装方法:**
- 重み非負制約 + 活性化関数の制約
- ペナルティ項: `λ * max(0, -∂f/∂x_j)²`
- Input convex neural networks (ICNN) の応用

### Phase 3: ガウス過程SFA (GPFrontierSFA)

**アプローチ:** GPyTorch を活用した確率的フロンティア推定
- カーネル: Matérn 5/2 or RBF（フロンティアの滑らかさに応じて選択）
- 単調性: 仮想微分観測 (virtual derivative observations)
- 利点: 自然な不確実性定量化、小〜中規模データに最適
- 誤差分解: カスタム尤度関数（GPyTorchのカスタムLikelihood）

### Phase 4: カーネルSFA (KernelSFA)

**アプローチ:** Fan, Li, Weersink (1996) / Kumbhakar et al. (2007)
- Nadaraya-Watson推定量 or 局所多項式回帰
- 局所最尤推定: 各点でカーネル重み付きSFA対数尤度を最大化
- バンド幅選択: leave-one-out クロスバリデーション

---

## 4. 共通コンポーネント

### 4.1 非効率性分布 (`distributions.py`)

```python
class InefficiencyDistribution(Protocol):
    def log_pdf(self, u: NDArray) -> NDArray: ...
    def cdf(self, u: NDArray) -> NDArray: ...
    def conditional_mean(self, epsilon: NDArray, sigma_v: float, sigma_u: float) -> NDArray: ...
    def conditional_mode(self, epsilon: NDArray, sigma_v: float, sigma_u: float) -> NDArray: ...

# 実装: HalfNormal, TruncatedNormal, Exponential
```

### 4.2 Shape制約 (`constraints.py`)

```python
class MonotonicityConstraint:
    """∂f/∂x_j ≥ 0 を強制（生産関数の単調増加性）"""

class ConcavityConstraint:
    """∂²f/∂x_j² ≤ 0 を強制（限界生産力逓減）"""
```

### 4.3 シミュレーションデータ (`simulator.py`)

モデル検証用のDGP（データ生成過程）:
- Cobb-Douglas DGP（ベースライン）
- Translog DGP
- 非線形フロンティアDGP（MLの優位性を示す）
- パネルデータDGP

```python
def simulate_sfa(
    n_obs: int,
    n_inputs: int,
    frontier_type: str,  # "cobb-douglas" | "translog" | "nonlinear"
    inefficiency_dist: str,  # "half-normal" | "truncated-normal"
    sigma_v: float,
    sigma_u: float,
    seed: int,
) -> SFADataset:
    """シミュレーションデータを生成"""
```

### 4.4 評価指標 (`metrics.py`)

| 指標 | 説明 |
|------|------|
| RMSE(TE) | 真のTE vs 推定TEのRMSE |
| Rank correlation | 効率性ランキングのSpearman相関 |
| Log-likelihood | 対数尤度 |
| AIC / BIC | モデル選択基準 |
| Coverage | 信頼区間のカバー率 |
| Frontier MSE | フロンティア関数の推定精度 |

---

## 5. 実装ロードマップ

### Phase 1: ParametricSFA + 基盤 (MVP)
- [ ] `BaseSFAEstimator` 基底クラス
- [ ] `InefficiencyDistribution` (half-normal, truncated-normal, exponential)
- [ ] `ParametricSFA` (Cobb-Douglas + half-normal)
- [ ] `simulator.py` (Cobb-Douglas DGP)
- [ ] 評価指標 (RMSE, rank correlation)
- [ ] ベンチマーク: pySFA / statsmodels との比較

### Phase 2: NNFrontierSFA
- [ ] 2段階NN-SFA (MLP frontier + parametric decomposition)
- [ ] 単調性制約 (penalty-based)
- [ ] 同時推定NN-SFA (custom loss)
- [ ] 非線形DGPでのベンチマーク

### Phase 3: GPFrontierSFA
- [ ] GPyTorchベースのフロンティア推定
- [ ] カスタムSFA尤度
- [ ] 効率性の事後分布

### Phase 4: KernelSFA + 比較
- [ ] 局所多項式SFA
- [ ] 全モデルの体系的比較
- [ ] 論文用の実験設計

---

## 6. 技術的課題と対策

### 6.1 誤差分解の識別問題

**問題:** MLフロンティアが柔軟すぎると、非効率性 u を吸収してしまう。

**対策:**
- Shape制約（単調性・凹性）でフロンティアの過適合を防ぐ
- 正則化パラメータのCV選択
- シミュレーション実験で分解精度を検証
- パネルデータの時間構造を活用

### 6.2 計算コスト

**問題:** GP (O(N³)), MCMC, 大規模NNの計算量。

**対策:**
- GP: 変分推論, inducing points (GPyTorchのScalableGP)
- NN: ミニバッチ学習, GPU活用 (PyTorch)
- 中規模データ (N < 10,000) をまず対象

### 6.3 経済学的解釈性の保持

**問題:** MLモデルはブラックボックスになりがち。

**対策:**
- 部分依存プロット (PDP)
- 弾力性推定: `∂ln f / ∂ln x_j` の計算
- 規模の経済性: 弾力性の和
- 入力ごとの限界効果の可視化

---

## 7. API設計例

```python
from ml_sfa.models import ParametricSFA, NNFrontierSFA
from ml_sfa.data import simulate_sfa

# シミュレーションデータ
data = simulate_sfa(n_obs=500, n_inputs=3, frontier_type="cobb-douglas",
                    inefficiency_dist="half-normal", sigma_v=0.1, sigma_u=0.2, seed=42)

# 伝統的SFA
model_param = ParametricSFA(frontier="cobb-douglas", inefficiency="half-normal")
model_param.fit(data.X, data.y)
te_param = model_param.efficiency(data.X, data.y)

# NN-SFA
model_nn = NNFrontierSFA(hidden_layers=[64, 32], monotonic=True,
                          inefficiency="half-normal", epochs=200)
model_nn.fit(data.X, data.y)
te_nn = model_nn.efficiency(data.X, data.y)

# モデル比較
from ml_sfa.evaluation import compare_models
results = compare_models(
    models={"Parametric": model_param, "NN": model_nn},
    X=data.X, y=data.y, true_te=data.te,
)
print(results.summary())
```

---

## 8. 参考文献

### 基礎
- Aigner, Lovell, Schmidt (1977). Formulation and Estimation of Stochastic Frontier Production Function Models. *J. Econometrics*
- Meeusen, van den Broeck (1977). Efficiency Estimation from Cobb-Douglas Production Functions with Composed Error. *Int. Econ. Rev.*
- Jondrow et al. (1982). On the Estimation of Technical Inefficiency in the SFA Model. *J. Econometrics*

### NN-SFA
- Pendharkar (2023). RBF Neural Network for Stochastic Frontier Analyses. *Neural Processing Letters*
- Kutlu (2024). A Machine Learning Approach to Stochastic Frontier Modeling. *Research Square*
- Tsionas, Parmeter, Zelenyuk (2023). Bayesian ANN for Frontier Efficiency Analysis. *J. Econometrics*
- Zhao (2024). xNN-SF: Explainable Neural Network Inspired by SFA. *ICIC 2024*

### ノンパラメトリック
- Fan, Li, Weersink (1996). Semiparametric Estimation of Stochastic Production Frontier Models. *JBES*
- Kumbhakar, Park, Simar, Tsionas (2007). Nonparametric Stochastic Frontiers: Local MLE. *J. Econometrics*
- Kuosmanen, Kortelainen (2012). StoNED. *J. Productivity Analysis*

### ツリー・その他
- Ferrara, Vidoli (2024). BART for SFA. *ECOSTA*
- Esteve et al. (2020). Efficiency Analysis Trees. *Expert Systems with Applications*
- Zheng et al. (2024). Robust Nonparametric SFA. *arXiv:2404.04301*
- GeMA (2025). Learning Latent Manifold Frontiers. *arXiv:2603.16729*
