# ThoughtComm 作業履歴

論文: "Thought Communication in Multiagent Collaboration" (arXiv:2510.20733, NeurIPS 2025)
リポジトリ: https://github.com/AUMEZAK/thoughtcomm

---

## Phase 1: 初期実装 (2026-02-15)

**コミット**: `f015a4e` Initial commit: ThoughtComm full reproduction

### 実施内容
- 論文の全10フェーズをゼロから実装
  1. `config.py` — ハイパーパラメータ設定
  2. `data/synthetic.py` — 合成データ生成（Laplace分布 + 線形混合）
  3. `data/math_loader.py`, `data/gsm8k_loader.py` — 評価データセットローダー
  4. `pipeline/debate.py` — マルチエージェント議論パイプライン
  5. `pipeline/hidden_states.py` — Hidden State 抽出
  6. `models/autoencoder.py` — Sparsity-Regularized AE（Jacobian L1ペナルティ）
  7. `pipeline/agreement.py` — Agreement-based Reweighting
  8. `models/prefix_adapter.py` — Prefix Adapter (Z̃ → prefix embedding)
  9. `evaluation/` — 評価メトリクス
  10. `notebooks/01-05` — Colab ノートブック5本

### アーキテクチャ決定
- モデル: Qwen-3-0.6B (hidden=1024) / Phi-4-mini-instruct (hidden=3072)
- 3エージェントが単一モデルを共有（逐次生成）
- Hidden state = 最終層・最終トークン
- Stochastic Jacobian推定（訓練時64行サンプリング）
- Full Jacobian: `torch.func.jacrev` + `vmap`（事後計算）

---

## Phase 2: Colab統合・バグ修正 (2026-02-15 〜 02-16)

**コミット**:
- `9e1837d` Rewrite README
- `b7ee41b` Add GitHub push functionality
- `025df5a` Fix critical bugs: cell order in notebooks and relative imports
- `a4518eb` Merge branch 'main'

### 実施内容
- README を包括的ドキュメントに書き直し
- 全ノートブックに GitHub push 機能を追加
- ノートブックのセル順序バグを修正
- 相対インポートの修正

---

## Phase 3: 合成実験の調査 (2026-02-27 〜 02-28)

**コミット**:
- `cc029cb` Speed up Notebook 1: vectorize Jacobian L1 + tune hyperparams
- `2f4f830` WIP: Increase jacobian_l1_weight for synthetic experiment
- `df9b900` Add CLAUDE.md session handoff document

### 問題
Notebook 01 の合成実験で、AE+L1 が期待通りの結果を出さない：
- dim=6 で MCC=0.21（期待値 >0.75）
- B matrix の復元精度が低い

### 調査内容 (Session 1-5)
1. **Session 1**: Jacobian L1 の vmap 実装を高速化。λ 調整を試行
2. **Session 2**: λ sweep、サンプル数増加、学習率調整を試行 → 効果なし
3. **Session 3**: 根本原因の特定
   - B_true の非ゼロ率が 66% → L1 で発見困難な密度
   - AE+L1 は block-sparse 構造を発見できない
   - 分離比 (separation ratio) = 1.0x → 構造的に識別不能
4. **Session 4**: ICA（Independent Component Analysis）を代替手法として発見
   - FastICA（dim ≤ 256）+ Picard（dim > 256）
   - `training/ica_solver.py` を新規作成
   - 合成データの `num_layers=1`（線形混合に限定）
5. **Session 5**: ICA でMCC > 0.97 を達成。ただし MCC permutation バグを発見

### 結論
AE+L1 は論文の合成実験には不適切。ICA が正しい手法。

---

## Phase 4: MCC Permutation バグ修正 (2026-03-02)

### 問題
`evaluation/synthetic_eval.py` の `compute_mcc` / `compute_mcc_fast`:
- `linear_sum_assignment` は `col_ind`（Z_hat → Z_true のマッピング）を返す
- 下流コードは `Z_hat[:, perm]` で使用 → Z_true → Z_hat のマッピングが必要
- 方向が逆

### 修正
```python
# 旧: return mcc, col_ind
# 新:
perm = np.argsort(col_ind)  # 逆置換
return mcc, perm
```

### 修正前後の比較 (dim=6, n_z=3)
| メトリクス | 修正前 | 修正後 |
|-----------|--------|--------|
| Per-component \|corr\| | 0.006 | 0.985 |
| R² diagonal | 0.28-0.33 | 0.967-0.983 |
| B matrix F1 | 0.65 | 1.000 |

---

## Phase 5: 全ノートブック一括修正 (2026-03-03)

**コミット**: `2e1bfe7` Fix AE normalization propagation, MCC permutation, and ICA solver

### 発見したクリティカルバグ: AE 正規化の未伝播

**問題**:
- `train_autoencoder()` は内部で H_train を正規化（mean/std）して訓練
- `SparsityRegularizedAE.encode()` は `self.encoder(H)` をそのまま返す（正規化なし）
- → Notebook 03 の B matrix、Notebook 04 の adapter 訓練、Notebook 05 の推論すべてで誤った Z_hat が計算される

**修正**: モデル内部に正規化を組み込み

#### 1. `models/autoencoder.py`
```python
# __init__ に register_buffer 追加
self.register_buffer('_norm_mean', torch.zeros(n_h))
self.register_buffer('_norm_std', torch.ones(n_h))
self.register_buffer('_has_norm', torch.tensor(False))

# 新メソッド
def set_norm_stats(self, mean, std):
    self._norm_mean.copy_(mean.detach())
    self._norm_std.copy_(std.detach().clamp(min=1e-8))
    self._has_norm.fill_(True)

def _normalize(self, H):
    if self._has_norm:
        return (H - self._norm_mean.to(H.device)) / self._norm_std.to(H.device)
    return H

# encode() を修正
def encode(self, H):
    return self.encoder(self._normalize(H))
```

**設計ポイント**:
- `_has_norm` は訓練中 `False` → 事前正規化済みデータをそのまま通す
- `set_norm_stats()` は訓練ループ終了後に呼ぶ → 以降の `encode()` が自動正規化
- `register_buffer` → `state_dict()` に含まれ、`load_state_dict()` で復元

#### 2. `training/train_autoencoder.py`
- `train_autoencoder()`: 訓練後に `model.set_norm_stats(H_mean, H_std)` 呼び出し
- `train_autoencoder_baseline()`: H_mean/H_std 追跡、`set_norm_stats()` 呼び出し、返り値を3値に変更

#### 3. Notebook 01 (`01_synthetic_experiment.ipynb`)
- Cell 6: baseline unpacking を 2→3 値に修正
- Cell 7: 手動正規化を除去（二重正規化防止）
- Cell 11 (MCC sweep): 手動正規化を除去

#### 4. Notebook 03 (`03_train_autoencoder.ipynb`)
- Cell 4: `train_autoencoder` の返り値を 2→3 値に修正
- Cell 9: `norm_stats.pt` を追加保存（後方互換用）

#### 5. Notebook 04 (`04_train_adapter.ipynb`)
- Cell 4: `strict=False` + 旧チェックポイント用 fallback

#### 6. Notebook 05 (`05_full_evaluation.ipynb`)
- Cell 3: `strict=False` + 旧チェックポイント用 fallback
- 新セル: GSM8K debate baseline 評価（Table 1 の "--" を実値に置換）
- 結果 dict に debate_gsm8k 結果を追加

#### 自動修正されたファイル（コード変更不要）
| ファイル | 箇所 | 理由 |
|---------|------|------|
| `training/train_adapter.py` L113 | `autoencoder.encode(H_t)` | encode() が自動正規化 |
| `pipeline/thought_comm.py` L74 | `self.ae.encode(H_t)` | encode() が自動正規化 |
| `notebooks/02_collect_hidden_states.ipynb` | — | AE 不使用 |

### 検証結果（5テスト全通過）
1. `_has_norm=False` 時に encode() が素通しすること
2. `state_dict()` → `load_state_dict()` で norm_stats が復元されること
3. 旧チェックポイント（`strict=False`）読み込みが動作すること
4. `train_autoencoder()` が norm_stats を正しく埋め込むこと
5. End-to-end パイプライン: ICA MCC=0.9586, R² diagonal > 0.8

---

## 新規ファイル一覧

| ファイル | 説明 |
|---------|------|
| `training/ica_solver.py` | FastICA/Picard ベースの ICA ソルバー |

---

## 現在の実行手順

1. **Notebook 01** (ローカル or Colab): 合成実験（ICA 検証）
2. **Notebook 02** (Colab GPU): Qwen-3-0.6B で Hidden State 収集
3. **Notebook 03** (Colab GPU): AE 訓練 + B matrix 計算
4. **Notebook 04** (Colab GPU): Prefix Adapter 訓練
5. **Notebook 05** (Colab GPU): 最終評価（Table 1 再現）

### Colab リンク（ブランチ: `claude/investigate-codebase-z2oid`）
- [01_synthetic_experiment](https://colab.research.google.com/github/AUMEZAK/thoughtcomm/blob/claude/investigate-codebase-z2oid/notebooks/01_synthetic_experiment.ipynb)
- [02_collect_hidden_states](https://colab.research.google.com/github/AUMEZAK/thoughtcomm/blob/claude/investigate-codebase-z2oid/notebooks/02_collect_hidden_states.ipynb)
- [03_train_autoencoder](https://colab.research.google.com/github/AUMEZAK/thoughtcomm/blob/claude/investigate-codebase-z2oid/notebooks/03_train_autoencoder.ipynb)
- [04_train_adapter](https://colab.research.google.com/github/AUMEZAK/thoughtcomm/blob/claude/investigate-codebase-z2oid/notebooks/04_train_adapter.ipynb)
- [05_full_evaluation](https://colab.research.google.com/github/AUMEZAK/thoughtcomm/blob/claude/investigate-codebase-z2oid/notebooks/05_full_evaluation.ipynb)

---

## 残課題

- [ ] Notebook 02-05 を Colab で実行して動作確認
- [ ] Qwen-3-0.6B での MATH/GSM8K 結果を Table 1 と比較
- [ ] Phi-4-mini-instruct での実験
- [ ] Fig 5/6（ラウンド別精度推移）は将来の拡張
- [ ] ブランチを main にマージ
