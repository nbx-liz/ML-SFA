# ML-SFA 仕様変更履歴 (Proposals / Decisions / Migrations)

本ファイルは ML-SFA における仕様変更の提案 (Proposal) と決定 (Decision) を時系列で記録する。
`change-gate.md` ルール (`~/.claude/rules/project/change-gate.md`) に従い、
公開 API・データ契約・アーキテクチャ・依存追加に関わる変更は本ファイルへの Proposal を経て確定する。

ID 体系: `P-XXXX` (Proposal), `D-XXXX` (Decision, P と同じ番号), `M-XXXX` (Migration)

---

## P-0001: フロンティア分析モジュール `ml_sfa.frontier` の追加

| 項目 | 値 |
|------|-----|
| 提案日 | 2026-05-07 |
| Status | Proposed |
| 種別 | Major scope addition (新規名前空間 + 新規依存 + 新規データ契約) |
| 関連ドキュメント | [`docs/research/deterministic_frontier_survey.md`](docs/research/deterministic_frontier_survey.md) (教科書級サーベイ), [`docs/glossary.md`](docs/glossary.md) (用語集) |

### 1. 目的 (Purpose)

ML-SFA は確率的フロンティア (SFA) と ML 拡張を主対象として開発されてきた。
本提案は **決定論的・ノンパラメトリック系のフロンティア分析手法**
(DEA, FDH, Order-m/α, COLS, CNLS, StoNED, EAT, RFEAT, Malmquist, DDF 等) を
新規名前空間 `ml_sfa.frontier` として網羅的に提供することを目的とする。

学習コスト最小化を全設計判断の指導原理とし、教科書サーベイ §3〜§9 と
実装クラスを 1:1 対応させる。

### 2. 影響範囲 (Impact)

| 項目 | 影響 |
|------|------|
| 公開 API | **追加のみ**。`ml_sfa.frontier.*` を新設。`ml_sfa.models.*` には変更なし |
| データ契約 | 新規 `Result` 型 `FrontierSummary`、新規 fitted attributes 規約 |
| 依存関係 | `scipy` (既存) を必須利用。`cvxpy` を**オプショナル extra `[qp]`** として追加 |
| ライセンス | MIT 維持。GPL-2/3 データの直接同梱は禁止 (`tests/data/external/download.py` 経由) |
| ドキュメント | サーベイ + glossary 新設、BLUEPRINT.md は SFA トラック特化のまま温存 |
| 既存テスト | 影響なし |

### 3. 互換性評価 (Compatibility)

- 既存 `BaseSFAEstimator` 系統には**いかなる変更もしない**
- `pyproject.toml` に `[project.optional-dependencies] qp = ["cvxpy>=1.5"]` を追加
- 既存 `[nn]` / `[bart]` と同じ extra パターンに揃う
- ユーザは `pip install ml-sfa[qp]` で QP ベース手法 (CNLS / StoNED) を利用可能
- ユーザが extra を入れない場合、CNLS / StoNED は import 時に明示的なエラーで案内 (既存 nn / bart と同じパターン)

### 4. 代替案検討 (Alternatives Considered)

| 代替案 | 却下理由 |
|--------|----------|
| 別パッケージ化 (`ml-sfa-frontier` リポジトリ分離) | 評価指標の共有が破綻、ユーザに 2 つの pip install を強要、開発者リソース分散 |
| 既存 `BaseSFAEstimator` を拡張して DEA/FDH を継承させる | DEA は transductive (`predict` 概念が SFA と異質)、`one class one job` 原則に違反 |
| `ml_sfa.dea` / `ml_sfa.fdh` 等手法ごとに名前空間分割 | サブパッケージ過多、共通評価ユーティリティの所属が不明確 |
| `ml_sfa.envelopment` 等別名 | サーベイの守備範囲 (DEA/FDH/StoNED/COLS 等) が単一概念で表せず |

→ **`ml_sfa.frontier` 単一名前空間が最も整合的**

### 5. 設計決定 (Decisions)

#### D-0001-01: クラス階層 (4 Family Base)

```
ml_sfa.models.BaseSFAEstimator               (既存、変更なし)

ml_sfa.frontier.BaseFrontierEstimator        (新規 ABC)
├── BaseEnvelopment       (DEA, FDH 系: lambdas/slacks/peers を持つ)
│   ├── CCR, BCC, Additive, SBM, FDH
│   ├── SuperCCR, SuperBCC, SuperSBM         (子クラス, Andersen-Petersen 1993)
│   └── UndesirableSBM                       (子クラス, Tone 2003)
├── BaseRobustFrontier    (確率成分あり、CI を返す)
│   ├── OrderM                                (独立, Cazals-Florens-Simar 2002)
│   └── OrderAlpha                            (独立, Aragon-Daouia 2005)
├── BaseFunctionalFrontier (predict(X) が定義可能)
│   ├── COLS, MOLS, TFA, DFA
│   ├── CNLS, StoNED
│   └── EAT, RFEAT
└── BaseDirectionalDistance (DDF 系)
    └── DDF
```

サーベイ章番号と完全 1:1 対応 (§3 → BaseEnvelopment, §4 → BaseRobustFrontier の一部, §5-§7 → BaseFunctionalFrontier, §9 → BaseDirectionalDistance)。

#### D-0001-02: 共通最小 IF

```python
class BaseFrontierEstimator(BaseEstimator, ABC):
    @abstractmethod
    def fit(self, X, Y=None, *, inputs=None, outputs=None) -> Self: ...

    @abstractmethod
    def efficiency(self, X=None, Y=None) -> NDArray | pd.Series: ...

    def summary(self) -> FrontierSummary:    # 共通実装 (D-0001-08)
        ...

    def score(self, X, Y, sample_weight=None) -> float:
        return float(np.mean(self.efficiency(X, Y)))
```

必須は `fit` / `efficiency` の 2 つ。`summary` / `score` は共通実装。

#### D-0001-03: DataFrame 統合

`fit(X, Y)` (ndarray) または `fit(df, inputs=[...], outputs=[...])` (DataFrame) を許容。

```python
def fit(self, X, Y=None, *, inputs=None, outputs=None) -> Self:
    if isinstance(X, pd.DataFrame):
        if Y is not None:
            raise ValueError("Pass DataFrame to X with inputs=/outputs=, not Y separately")
        if inputs is None or outputs is None:
            raise ValueError("inputs= and outputs= required when X is DataFrame")
        ...
    return self._fit_impl(X_arr, Y_arr)
```

**入出力対称性 (規約):** 入力が DataFrame ならば、`efficiency()` / `peers()` 等の戻り値は
DMU index を保持した `pd.Series` / `pd.DataFrame`。入力が ndarray ならば ndarray を返す。
**この規約は全 `BaseFrontierEstimator` 継承クラスで強制**する。

#### D-0001-04: 効率性スコアの値域

原値で返す。正規化はしない。各クラスは fitted attribute `efficiency_range_: tuple[float, float]` を必須公開する。

| クラス | 値域 | 効率的の定義 |
|--------|------|---------------|
| CCR / BCC (input-oriented) | (0, 1] | 1 |
| CCR / BCC (output-oriented) | [1, ∞) | 1 |
| SBM | (0, 1] | 1 |
| Additive | [0, ∞) | 0 (スラック和) |
| FDH | (0, 1] (input) | 1 |
| OrderM (input) | (0, ∞) | ≤ 1 で効率的 |
| OrderAlpha | (0, ∞) | 同上 |
| Super-eff | (0, ∞) | 1 を超えうる |
| COLS / MOLS | (0, 1] (TE) | 1 |

各クラスの docstring 冒頭に「効率値の範囲と効率的の定義」を必ず明記する。

#### D-0001-05: `predict(X)` と `project(X, Y)`

**`BaseFunctionalFrontier` のみ `predict(X)` を提供** (単一産出専用、便利関数)。
**全 `BaseFrontierEstimator` 継承クラスが `project(X, Y)` を提供** (普遍、多産出 OK)。

```python
class BaseFunctionalFrontier(BaseFrontierEstimator):
    def predict(self, X) -> NDArray:
        """Single-output frontier value f̂(X). Multi-output → use project."""

class BaseFrontierEstimator:
    def project(self, X, Y) -> tuple[NDArray, NDArray]:
        """Project (X, Y) onto fitted frontier. Universal."""
```

`BaseEnvelopment.project()` は LP 再計算、`BaseFunctionalFrontier.project()` は関数値 + 直交方向への射影、`BaseRobustFrontier.project()` は m-trim サンプル経由。

#### D-0001-06: Wrapper vs 子クラス vs 独立クラス

ルール:

```
1. base estimator のロジックが完全に再利用可能で、
   外側でラップするだけで成立 → Wrapper
2. base モデルの一部 (LP 制約 1 行 / 引数 1 個) を変更するだけ → 子クラス
3. 理論的定式化が独立で convergence/仮定が異なる → 独立クラス
4. 複数 base estimator を時系列/階層的に組合せる → 独立クラス + factory
```

| 手法 | 形式 |
|------|------|
| BootstrapDEA | Wrapper (任意の `BaseEnvelopment` を受ける) |
| CrossEfficiency | Wrapper (任意の radial DEA を受ける) |
| SuperCCR / SuperBCC / SuperSBM | 子クラス |
| UndesirableSBM | 子クラス (`SBM`) |
| OrderM / OrderAlpha | 独立クラス |
| MalmquistIndex | 独立クラス + factory 引数 |
| HicksMoorsteen | 独立クラス + factory 引数 |
| NetworkDEA / TwoStageDEA | 独立クラス |
| DDF | 独立クラス (BaseDirectionalDistance) |

#### D-0001-07: Wrapper の base estimator 受入規約

Wrapper クラス (`BootstrapDEA`, `CrossEfficiency`) は **fit 済みインスタンス**を受け取る:

```python
base = BCC(orientation="input")
base.fit(X, Y)
boot = BootstrapDEA(base_estimator=base, n_bootstrap=2000, random_state=42)
boot.fit(X, Y)        # base からの再 fit を内部で実施
```

理由:
- ユーザが「base が確かに動くこと」を先に検証してから wrapper に渡せる
- wrapper の fit は内部で base の clone を生成しブートストラップ標本ごとに再 fit する
- `sklearn.base.clone(base_estimator)` で hyperparameter のみコピー

#### D-0001-08: Malmquist / Hicks-Moorsteen の API

パネル形式の fit を採用:

```python
mpi = MalmquistIndex(base_estimator=BCC(orientation="input"))
mpi.fit(X_panel, Y_panel, time_index=t)
# X_panel: (n_dmus * n_periods, n_inputs) または DataFrame で time 列を持つ
# Y_panel: 同上
# time_index: 各観測の時点 array, または DataFrame の列名

mpi.malmquist_                  # (n_dmus, n_periods - 1) の MPI
mpi.efficiency_change_          # キャッチアップ効果
mpi.frontier_shift_             # 技術前進効果
```

理由: 通常のパネル分析と同じデータ形式を使え、複数期間 (T > 2) を一気に処理できる。

#### D-0001-09: `summary()` の共通実装 + Hook

Template Method パターン:

```python
@dataclass
class FrontierSummary:
    method: str
    n_dmus: int
    n_inputs: int
    n_outputs: int
    n_efficient: int
    efficiency_mean: float
    efficiency_min: float
    efficiency_max: float
    efficiency_range: tuple[float, float]
    extras: dict[str, Any]                    # 手法固有の補助情報

    def __repr__(self) -> str: ...
    def to_dict(self) -> dict: ...
    def to_dataframe(self) -> pd.DataFrame: ...

class BaseFrontierEstimator:
    def summary(self) -> FrontierSummary:
        return FrontierSummary(
            method=self._method_name(),
            n_dmus=self.n_dmus_,
            ...
            extras=self._summary_extras(),
        )

    def _summary_extras(self) -> dict[str, Any]:
        return {}    # override in family / concrete classes
```

各 family base / 具象クラスは `_summary_extras()` のみオーバーライドして固有情報 (Bootstrap CI, σ_u/σ_v, peers 統計など) を追加する。

#### D-0001-10: ハイパーパラメータ 5 分類

全 docstring で以下のタグを使用:

| タグ | 意味 | 既定値方針 |
|------|------|-----------|
| `[STRUCTURAL]` | モデル定義の一部 (orientation, returns, weights) | 慎重に決め CV 対象外 |
| `[INFERENCE]` | 統計的推論 (n_bootstrap, bandwidth, confidence_level) | 理論的既定値、通常変更不要 |
| `[ROBUSTNESS]` | 外れ値耐性 (m, alpha, quantile_loss_tau) | `auto` モード or ユーザ指定 |
| `[ML-HP]` | ML 系 (max_depth, n_estimators, learning_rate) | CV 対象、frontier-aware scorer |
| `[COMPUTATIONAL]` | 数値計算 (solver, tol, max_iter) | `solver_options=` dict にカプセル化 |

#### D-0001-11: チューニング・モジュール

DEA 系の標準 K-fold CV は理論的に意味を持たない。専用モジュール `ml_sfa.frontier.tuning` を新設:

- `bootstrap_bandwidth_selection(estimator, X, Y, ...)` — Simar-Wilson 流
- `frontier_loocv(estimator, X, Y, ...)` — CNLS/StoNED 用
- `frontier_score(...)` — sklearn scorer 互換 (片側残差)

EAT / RFEAT は標準 sklearn `GridSearchCV` 互換 (`check_estimator` 完全準拠)。

#### D-0001-12: sklearn 互換性レベル

- `BaseFrontierEstimator` は `sklearn.base.BaseEstimator` を継承 (`get_params`/`set_params` 対応)
- `check_estimator` 完全準拠: `EAT`, `RFEAT`, `CNLS`, `StoNED` のみ
- 上記以外 (DEA/FDH/Order-m 系) は緩い準拠 (transductive のため)

#### D-0001-13: ディレクトリ構成

```
src/ml_sfa/frontier/
├── __init__.py
├── _base.py                # BaseFrontierEstimator + family bases
├── _summary.py             # FrontierSummary dataclass
├── _validation.py          # 入力検証 (負値/NaN/ゼロ)
├── envelopment/
│   ├── __init__.py
│   ├── ccr.py              # CCR, SuperCCR
│   ├── bcc.py              # BCC, SuperBCC
│   ├── additive.py
│   ├── sbm.py              # SBM, SuperSBM, UndesirableSBM
│   └── fdh.py
├── robust/
│   ├── __init__.py
│   ├── order_m.py
│   └── order_alpha.py
├── functional/
│   ├── __init__.py
│   ├── cols.py             # COLS, MOLS
│   ├── tfa.py
│   ├── dfa.py
│   ├── cnls.py
│   ├── stoned.py
│   ├── eat.py
│   └── rfeat.py
├── ddf/
│   ├── __init__.py
│   └── directional.py
├── wrappers/
│   ├── __init__.py
│   ├── bootstrap.py        # BootstrapDEA
│   └── cross_efficiency.py
├── productivity/
│   ├── __init__.py
│   ├── malmquist.py
│   ├── hicks_moorsteen.py
│   └── fare_primont.py
├── network/
│   ├── __init__.py
│   ├── network_dea.py
│   └── two_stage.py
└── tuning/
    ├── __init__.py
    ├── bootstrap.py
    ├── loocv.py
    └── scoring.py

src/ml_sfa/data/
└── published.py            # 公開データセットローダ (load_charnes1981 等)

tests/
├── unit/frontier/
├── integration/frontier/
└── data/published/         # 公開データの CSV + REFERENCE.md
```

### 6. 受入基準 (Acceptance Criteria)

各 Phase の PR で以下を満たすこと:

| # | 基準 | 検証方法 |
|---|------|----------|
| AC-1 | 単体テスト 80%+ カバレッジ | `pytest --cov` |
| AC-2 | サーベイ §11.2 のシミュレーション DGP で点推定 RMSE が文献既知レート以内 | `tests/unit/frontier/test_*_simulation.py` |
| AC-3 | サーベイ §11.3 の公開データ公表値再現 (誤差 < 1e-3) | `tests/integration/frontier/test_*_reproduction.py` |
| AC-4 | クロスソフトウェア整合 (pyStoNED / R Benchmarking との数値一致) | optional テスト、`pytest.importorskip` 付き |
| AC-5 | sklearn `check_estimator` 完全準拠 (ML 系のみ) | `tests/unit/frontier/test_sklearn_compatibility.py` |
| AC-6 | プロパティテスト (FDH ≤ BCC ≤ CCR、unit invariance、translation invariance) | hypothesis ベース |
| AC-7 | 効率値域・効率的の定義が docstring 冒頭に記載 | doc-lint (将来) |
| AC-8 | `[STRUCTURAL]/[INFERENCE]/...` タグが全パラメータ docstring に付与 | doc-lint (将来) |
| AC-9 | `efficiency()` の入出力対称性 (D-0001-03) | `tests/unit/frontier/test_io_symmetry.py` |
| AC-10 | `make ci` (ruff + mypy + pytest) がグリーン | CI |

### 7. リリース計画 (参考、Phase 別 Proposal で確定)

| Phase | 範囲 | 想定 PR 数 | バージョン |
|-------|------|-----------|-----------|
| F1 | 基底 + DEA-CCR/BCC/Additive/SBM | 3-5 | v0.2.0 |
| F2 | FDH + Order-m / Order-α | 2-3 | v0.3.0 |
| F3 | COLS / MOLS / TFA / DFA | 2 | v0.4.0 |
| F4 | CNLS / StoNED (`[qp]` extra 追加) | 3-4 | v0.5.0 |
| F5 | EAT / RFEAT | 2-3 | v0.6.0 |
| F6 | BootstrapDEA / CrossEfficiency / Simar-Wilson 二段階 | 2-3 | v0.7.0 |
| F7 | Malmquist / Hicks-Moorsteen / DDF / Undesirable / NetworkDEA | 3-4 | v0.8.0 |

各 Phase は独立 Proposal `P-0002`〜`P-0008` で詳細設計を確定する。

### 8. 未解決事項 (Open Issues)

- O-1: GPL-2 ライセンスのデータ (R `Benchmarking` 系) を `tests/data/published/` に CSV として同梱する場合のライセンス整理 → 別 Proposal `P-0009` (案) で扱う
- O-2: 大規模 LP (n > 1000) での scipy `linprog` HiGHS の収束性検証 → F1 着手時に実測
- O-3: cvxpy のバージョン互換性 (1.5+ で API 変更あり) → F4 着手時に依存固定範囲を確定
- O-4: `frontier_score` (片側残差スコア) の正式定義 → F5 着手時に Daouia et al. の Polynomial loss を参考に確定

### 9. Status 履歴

| 日付 | Status | 備考 |
|------|--------|------|
| 2026-05-07 | Proposed | 初稿作成。a/b/c/d + e/f/g/h + i/j/k 全合意済み |

---
