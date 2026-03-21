# 同時推定 NN-SFA 詳細設計

**作成日:** 2026-03-22

---

## 1. 概要

ニューラルネットワークで生産フロンティアを推定し、SFA対数尤度をカスタム損失関数として
フロンティアと誤差分解を**同時に**最適化する手法。

2段階法（NN→SFA）と異なり、フロンティア推定と誤差分解が単一の最適化問題として解かれるため、
NNがフロンティアに非効率性 u を吸収する問題を構造的に軽減する。

### 先行研究

| 文献 | 手法 | 特徴 |
|------|------|------|
| Tsionas, Parmeter, Zelenyuk (2023) | ベイズNN (正規混合) | MCMC推論、分布仮定フリー |
| Pendharkar (2023) | RBF-NN + MLE | 2層RBF、K-means初期化 |
| Zhao (2024) xNN-SF | 構造化3サブネット | 単調性をアーキテクチャで保証 |

本設計では **勾配ベースの同時MLE推定**（PyTorch実装）を主軸とする。
ベイズNNアプローチ（Tsionas et al.）は計算コストが高く、まず頻度論的アプローチで基盤を構築する。

---

## 2. 数学的定式化

### 2.1 モデル

```
y_i = NN(x_i; θ) + v_i - u_i
```

| 記号 | 定義 |
|------|------|
| `NN(x_i; θ)` | NNによるフロンティア関数。パラメータ θ |
| `v_i ~ N(0, σ_v²)` | 対称ノイズ |
| `u_i ~ N⁺(0, σ_u²)` | 片側非効率性 (half-normal) |
| `ε_i = y_i - NN(x_i; θ) = v_i - u_i` | 合成誤差 |

### 2.2 対数尤度

合成誤差 ε = v - u の密度関数は畳み込みにより:

```
f(ε_i) = (2/σ) · φ(ε_i/σ) · Φ(-ε_i·λ/σ)
```

対数尤度:

```
ln L(θ, σ_v, σ_u) = Σ_i [ ln(2/σ) + ln φ(ε_i/σ) + ln Φ(-ε_i·λ/σ) ]
                   = const - N·ln σ + Σ_i [ -ε_i²/(2σ²) + ln Φ(-ε_i·λ/σ) ]
```

ここで:
- `σ² = σ_v² + σ_u²`
- `λ = σ_u / σ_v`
- `φ(·)` = 標準正規PDF
- `Φ(·)` = 標準正規CDF

### 2.3 推定パラメータ

同時に推定されるパラメータ:

| パラメータ | 説明 | 制約 |
|-----------|------|------|
| `θ` | NNの重み・バイアス | 単調性制約（後述） |
| `σ_v` | ノイズの標準偏差 | `σ_v > 0` |
| `σ_u` | 非効率性の標準偏差 | `σ_u > 0` |

`σ_v`, `σ_u` の正値制約は対数パラメータ化で処理:
```
σ_v = exp(log_σ_v),  σ_u = exp(log_σ_u)
```

### 2.4 効率性推定

推定完了後、Jondrow et al. (1982) の条件付き期待値:

```
E[u_i | ε_i] = σ_* · [ φ(μ_*/σ_*) / Φ(μ_*/σ_*) + μ_*/σ_* ]
```

ここで:
```
μ_* = -ε_i · σ_u² / σ²
σ_*² = σ_u² · σ_v² / σ²
```

技術的効率性:
```
TE_i = exp(-E[u_i | ε_i])
```

NNフロンティアの場合も JLMS 公式自体は変わらない。
残差 `ε_i = y_i - NN(x_i; θ̂)` と推定済み `σ̂_u`, `σ̂_v` を代入するだけである。

---

## 3. 勾配の導出

### 3.1 NNパラメータ θ に対する勾配

損失関数 `-ln L` の θ に対する勾配:

```
∂(-ln L)/∂θ = Σ_i [ ε_i/σ² + (λ/σ) · φ(z_i)/Φ(-z_i) ] · ∂NN(x_i;θ)/∂θ
```

ここで `z_i = ε_i · λ / σ`。

- `∂NN/∂θ`: PyTorch autograd が自動計算
- `φ(z_i)/Φ(-z_i)`: **逆ミルズ比** (inverse Mills ratio)。`Φ(-z_i) → 0` のとき発散

### 3.2 分散パラメータに対する勾配

`log_σ_v`, `log_σ_u` に対する勾配も autograd で自動計算される。
手動導出は不要だが、数値安定性のため以下を確認:

```
∂(-ln L)/∂(log_σ_u) = ∂(-ln L)/∂σ_u · σ_u  (chain rule)
```

### 3.3 数値安定性

**`ln Φ(x)` の計算:**

| `x` の範囲 | 計算方法 |
|------------|---------|
| `x > -5` | `torch.log(torch.special.ndtr(x))` で十分安定 |
| `x < -5` | `torch.special.log_ndtr(x)` を使用（漸近展開ベース） |
| 全範囲 | **`torch.special.log_ndtr(x)` を常に使用**（推奨） |

PyTorch の `log_ndtr` は内部で `log(erfc(-x/√2)/2)` の漸近展開を用いて
極端な負の値でも数値的に安定。

**逆ミルズ比のクリッピング:**
```python
mills = torch.exp(torch.special.log_ndtr(z) - torch.special.log_ndtr(-z))
# または直接:
mills = phi(z) / Phi(-z)
mills = torch.clamp(mills, max=1e6)  # 発散防止
```

---

## 4. ネットワークアーキテクチャ

### 4.1 基本構成

```
入力 x ∈ R^p
    │
    ▼
[Linear(p, h1)] → [Softplus] → [Linear(h1, h2)] → [Softplus] → [Linear(h2, 1)]
    │                                                                │
    │           MLP フロンティアネットワーク                           │
    │                                                                ▼
    │                                                          f̂(x) ∈ R
    │
    ├── log_σ_v ∈ R  (learnable scalar parameter)
    └── log_σ_u ∈ R  (learnable scalar parameter)
```

### 4.2 推奨ハイパーパラメータ

| パラメータ | 推奨値 | 根拠 |
|-----------|--------|------|
| 隠れ層数 | 2 | 万能近似に十分。深すぎると過適合 |
| 隠れ層幅 | 64, 32 | 入力次元の10倍程度。SFAの典型的入力(2-10次元)に対して十分 |
| 活性化関数 | Softplus | 単調増加・微分可能。ReLU は微分不連続 |
| 出力活性化 | なし (Linear) | フロンティア出力は非有界 |
| ドロップアウト | なし | SFAの少〜中規模データでは正則化はL2で十分 |

### 4.3 単調性制約

生産関数の経済学的要請: `∂f/∂x_j ≥ 0`（全投入について産出は非減少）

**方法 A: 重み非負制約（アーキテクチャレベル）**

全ての重み行列を非負に制約し、単調増加活性化関数を使用:

```python
class MonotonicMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims):
        ...
        # 重みは生の値で保持し、forward 時に softplus で非負化
        self.raw_weights = nn.ParameterList([...])

    def forward(self, x):
        for raw_w, bias in zip(self.raw_weights, self.biases):
            w = F.softplus(raw_w)      # W ≥ 0 を保証
            x = F.softplus(x @ w + bias)  # 単調増加活性化
        return x
```

原理: 非負重み + 単調増加活性化 の合成は単調増加（連鎖律）。

**方法 B: ペナルティベース（ソフト制約）**

```
Loss = -ln L + λ_mono · Σ_i Σ_j max(0, -∂NN/∂x_j(x_i))²
```

- `∂NN/∂x_j` は autograd で計算
- `λ_mono` を徐々に増加（penalty annealing）
- 利点: アーキテクチャの自由度を維持
- 欠点: 単調性の厳密な保証はない

**推奨:** 方法 A（重み非負制約）を主手法とし、方法 B を比較実験用に用意。

---

## 5. 最適化戦略

### 5.1 ウォームスタート初期化

```
Phase 0 (pre-training):
  MSE損失で NN を学習 → OLS 的なフロンティア初期推定
  OLS残差の分散から σ_v, σ_u を初期化:
    σ̂² = Var(OLS残差)
    log_σ_v ← ln(√(σ̂²/2))
    log_σ_u ← ln(√(σ̂²/2))

Phase 1 (SFA fine-tuning):
  損失関数を -ln L (SFA対数尤度) に切り替え
  Phase 0 の重みから fine-tune
```

ウォームスタートが重要な理由:
- SFA対数尤度は非凸で、ランダム初期化からでは局所最適に捕まりやすい
- MSE事前学習がフロンティア近傍の合理的な初期点を提供

### 5.2 オプティマイザ

| オプティマイザ | 用途 | 設定 |
|---------------|------|------|
| **Adam** | Phase 0 (MSE) | lr=1e-3, weight_decay=1e-4 |
| **L-BFGS** | Phase 1 (SFA) | lr=1.0, max_iter=20, line_search='strong_wolfe' |

L-BFGS を Phase 1 で推奨する理由:
- 2次情報を利用し、SFA尤度の曲率に適応
- SFAの典型的なデータサイズ (N < 10,000) ではフルバッチが実用的
- 計量経済学的推定では Adam より収束が安定する傾向

### 5.3 収束判定

```
|ln L^(t) - ln L^(t-1)| / |ln L^(t-1)| < 1e-6  (相対変化)
かつ
||θ^(t) - θ^(t-1)||₂ < 1e-6  (パラメータ変化)
```

### 5.4 多重初期化

非凸最適化のため、複数の初期値で推定を実行:
- 5-10 回の異なるランダムシードで推定
- 最大対数尤度を達成した結果を採用
- 結果のばらつきが大きい場合はモデル識別の問題を示唆

---

## 6. 2段階法との比較

### 6.1 理論的差異

| 観点 | 2段階法 | 同時推定 |
|------|---------|---------|
| **フロンティア推定** | MSE最小化（OLS的） | SFA尤度最大化 |
| **誤差分解** | 残差に対して事後的に適用 | フロンティアと同時に推定 |
| **u の吸収問題** | NNが u を吸収するリスクあり | 尤度構造が u をフロンティアから分離 |
| **一貫性** | Stage 1 の仮定と Stage 2 が矛盾 | 単一の尤度関数で一貫 |
| **計算コスト** | 低（各 stage が独立） | 中〜高（非凸最適化） |

Wang & Schmidt (2002) は2段階推定で非効率性決定要因のパラメータに**深刻な下方バイアス**が
生じることをモンテカルロ実験で示した。同時推定はこの問題を回避する。

### 6.2 u 吸収問題の直感的説明

2段階法:
```
Stage 1: min_θ Σ(y_i - NN(x_i;θ))²
→ NNは y の条件付き平均 E[y|x] = f(x) - E[u] を学習
→ 非効率性の期待値 E[u] がフロンティアに吸収される
→ 残差 ε̂ の分布が歪まず、u の推定が不正確に
```

同時推定:
```
max_{θ,σ_v,σ_u} ln L(θ, σ_v, σ_u)
→ 尤度関数が ε の非対称性（歪度）をペナルティとして作用
→ NNは f(x) を（E[u] を含まない）真のフロンティアに近づけるよう誘導
→ 誤差分解がより正確に
```

### 6.3 同時推定が失敗するケース

1. **少量データ (N < 300)**: パラメータ過多による過適合。パラメトリック SFA が有利
2. **真のフロンティアがパラメトリック**: 正しい関数形を仮定したパラメトリック SFA のほうが効率的（分散が小さい）
3. **λ → 0 (非効率性が小さい)**: u の寄与が小さいとき、SFA 尤度の非対称項が弱く、フロンティアとノイズの分離が困難
4. **局所最適**: 非凸最適化のため、初期化に依存。多重初期化で緩和

---

## 7. 分布の拡張

### 7.1 対応する非効率性分布

| 分布 | u の密度 | 合成誤差の対数尤度 |
|------|---------|-------------------|
| Half-normal | `(2/σ_u)·φ(u/σ_u)` | `ln(2/σ) + ln φ(ε/σ) + ln Φ(-ελ/σ)` |
| Exponential | `(1/σ_u)·exp(-u/σ_u)` | `-ln σ_u + ε/σ_u + σ_v²/(2σ_u²) + ln Φ(-(ε/σ_v + σ_v/σ_u))` |
| Truncated normal | `φ((u-μ)/σ_u) / Φ(μ/σ_u)` | より複雑な閉じた形 |

各分布の `ln L` と JLMS 条件付き期待値はそれぞれ異なるが、
PyTorch autograd によりどの分布でも勾配は自動計算される。

### 7.2 異分散性拡張

非効率性分布のパラメータを共変量の関数にする:
```
σ_u(z_i) = exp(NN_u(z_i; θ_u))  (Battese-Coelli 1995 のNN拡張)
σ_v(z_i) = exp(NN_v(z_i; θ_v))
```

これにより企業特性や環境変数が非効率性の大きさに影響することをモデル化できる。

---

## 8. 実装上の注意点

### 8.1 PyTorch 実装の骨格

```python
class JointNNSFA(nn.Module):
    def __init__(self, in_dim, hidden_dims, monotonic=True):
        super().__init__()
        self.frontier_net = MonotonicMLP(in_dim, hidden_dims) if monotonic else MLP(in_dim, hidden_dims)
        self.log_sigma_v = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_u = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.frontier_net(x).squeeze(-1)

    def sfa_nll(self, x, y):
        f_hat = self.forward(x)
        eps = y - f_hat
        sigma_v = torch.exp(self.log_sigma_v)
        sigma_u = torch.exp(self.log_sigma_u)
        sigma = torch.sqrt(sigma_v**2 + sigma_u**2)
        lam = sigma_u / sigma_v
        z = eps * lam / sigma
        nll = -torch.sum(
            -torch.log(sigma)
            - 0.5 * (eps / sigma)**2
            + torch.special.log_ndtr(-z)
        )
        return nll
```

### 8.2 学習ループの骨格

```python
# Phase 0: MSE pre-training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(pretrain_epochs):
    loss = F.mse_loss(model(X), y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

# Phase 1: SFA fine-tuning
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20)
for epoch in range(finetune_epochs):
    def closure():
        optimizer.zero_grad()
        loss = model.sfa_nll(X, y)
        loss.backward()
        return loss
    optimizer.step(closure)
```

### 8.3 検証すべき事項

- [ ] シミュレーションDGPで真の TE との RMSE を計測
- [ ] パラメトリック SFA との比較（正しい関数形 DGP / 誤った関数形 DGP）
- [ ] 2段階 NN-SFA との比較（同一アーキテクチャで推定方法のみ変更）
- [ ] 単調性制約の有無による影響
- [ ] サンプルサイズ (N = 200, 500, 1000, 5000) での挙動
- [ ] σ_u / σ_v 比（λ）の異なるDGPでの分解精度

---

## 9. 参考文献

- Tsionas, M.G., Parmeter, C.F., Zelenyuk, V. (2023). Bayesian Artificial Neural Networks for Frontier Efficiency Analysis. *Journal of Econometrics*, 236(2), 105491.
- Pendharkar, P.C. (2023). A Radial Basis Function Neural Network for Stochastic Frontier Analyses. *Neural Processing Letters*, 55, 4995-5013.
- Zhao, X. (2024). xNN-SF: An Explainable Neural Network Inspired by SFA. *ICIC 2024*.
- Kutlu, L. (2024). A Machine Learning Approach to Stochastic Frontier Modeling. *Research Square*.
- Wang, H.J., Schmidt, P. (2002). One-Step and Two-Step Estimation of the Effects of Exogenous Variables on Technical Efficiency Levels. *Journal of Productivity Analysis*, 18, 129-144.
- Jondrow, J. et al. (1982). On the Estimation of Technical Inefficiency. *Journal of Econometrics*, 19(2-3), 233-238.
- Dugas, C. et al. (2009). Incorporating Functional Knowledge in Neural Networks. *JMLR*, 10, 1239-1262.
