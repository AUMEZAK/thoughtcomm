# Notebook 02・03 実行結果と課題 (2026-03-04)

## Notebook 02: Hidden State収集 — 完了

**コミット**: `b4c37ba`
**結果ファイル**: `results/02_hidden_states_summary_Qwen3-0.6B.json`

### 結果
| データセット | サンプル数 | 形状 | Mean | Std |
|---|---|---|---|---|
| MATH Level-3 | 200 | (200, 3072) | -0.118 | 4.133 |
| GSM8K | 200 | (200, 3072) | -0.171 | 4.350 |

- 3072 = 3 agents × 1024 (Qwen3-0.6B hidden_size)
- NaN/Inf なし、正常
- GSM8Kの方がstdがやや大きい
- チェックポイント: Google Drive `/content/drive/MyDrive/thoughtcomm_checkpoints/`

### ArrowKeyError修正
- **コミット**: `1cdc84d`
- **原因**: cell-1の`sys.modules`クリーンアップで`datasets`と`pyarrow`を削除していた
- pyarrowのC拡張型（Array2DExtensionType）はPythonモジュール削除後も型登録が残る
- 再importで重複登録エラー
- **修正**: クリーンアップ対象からdatasets/pyarrowを除外（02〜05全ノートブック）

---

## Notebook 03: AE訓練 + B行列 — 完了だが深刻な問題あり

**コミット**: `16976a4`
**結果ファイル**: `results/03_autoencoder_summary_Qwen3-0.6B.json`
**画像**: `results/ae_training_curves_Qwen3-0.6B.png`, `results/b_matrix_Qwen3-0.6B.png`

### AE訓練結果
| メトリクス | 値 |
|---|---|
| 最終 rec_loss | 13807 |
| 最終 jac_loss | 0.025 |
| 最終 total_loss | 13807 |
| エポック数 | 200 |

### B行列
| メトリクス | 値 |
|---|---|
| 形状 | (3072, 1024) |
| スパーシティ | 1.6% (= 98.4%非ゼロ) |
| 非ゼロ要素 | 3,095,310 / 3,145,728 |
| alpha=0 (どのエージェントにも無関係) | 0次元 |
| alpha=1 (1エージェントのみ) | 0次元 |
| alpha=2 (2エージェント合意) | 0次元 |
| alpha=3 (全エージェント合意) | 1024次元 (100%) |

### 問題1: 訓練不安定（爆発）
- epoch 0〜150: rec_lossが~1から~0.1へ順調に低下
- epoch ~150: lossが突然10^7に爆発
- epoch 180〜200: 10^4程度まで回復するが、epoch 150以前のレベルには戻らない
- **最終rec_loss=13807は爆発後の値**であり、正常な収束結果ではない

**原因分析**:
- gradient clippingなし → 勾配爆発を防げない
- LRスケジュールなし → lr=1e-3が後半で高すぎる
- early stoppingなし → 爆発後も走り続け、best modelが失われる

### 問題2: B行列が全結合（スパーシティなし）
- 98.4%が非ゼロ → ほぼ全結合
- 全1024次元がalpha=3 → agreement reweightingは全次元同じ重み → 意味なし
- 合成実験（Notebook 01）で発見したL1の限界と一致

**原因分析**:
- jacobian_l1_weight=0.01だが、jac_lossは~0.003
- 実効的寄与 = 0.01 × 0.003 = 0.00003 → rec_loss ~0.1に対して無視できる
- L1圧力がほぼゼロのため、スパーシティが誘導されない

### 使用した設定値
```python
# config.py からの設定
n_h = 3072          # 3 agents × 1024
n_z = 1024          # latent dimension
ae_hidden = 2048    # hidden layer width
ae_num_layers = 3   # encoder/decoder depth
ae_lr = 1e-3        # learning rate (固定、スケジュールなし)
ae_epochs = 200
ae_batch_size = 64  # 200サンプル / 64 = ~3 batches/epoch
jacobian_l1_weight = 0.01   # lambda
jacobian_sample_rows = 64   # stochastic Jacobian sampling
jacobian_threshold = 0.01   # B行列の二値化閾値
```

### train_autoencoder.pyの問題点
1. gradient clippingなし
2. LRスケジュールなし（200 epoch全てlr=1e-3固定）
3. early stoppingなし
4. best model保存なし（最後のモデル=爆発後がそのまま使われる）

---

## 修正プラン: AE訓練安定化 (Step 1)

### 目的
訓練爆発を防ぎ、best modelを確実に保存する。
jacobian_l1_weightは変更しない（安定化の効果を単独で確認するため）。

### 修正ファイル

#### 1. `training/train_autoencoder.py`
- **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), config.ae_grad_clip)` を `optimizer.step()` の前に追加
- **Cosine Annealing LR**: `CosineAnnealingLR(optimizer, T_max=config.ae_epochs, eta_min=config.ae_lr_min)`、各epoch末に `scheduler.step()`
- **Early stopping + best model保存**: best_loss / best_state_dict を追跡、patience epochs改善なしで訓練終了、終了時にbest modelを復元
- **verbose出力**: 現在のLRを追加表示

#### 2. `configs/config.py`
- `ae_grad_clip: float = 1.0` — gradient clipping max norm
- `ae_lr_min: float = 1e-5` — cosine annealing 最小LR
- `ae_patience: int = 30` — early stopping patience

#### 3. Notebook 03 — 変更不要
`train_autoencoder()` のAPIは変わらない（返り値: model, loss_history, norm_stats）。

### jacobian_l1_weightについて
現状の実効寄与≈0.00003はrec_loss≈0.1の0.03%で、ほぼ無意味。
しかし安定化と同時に変更すると効果の切り分けができない。
**安定化後にB行列が改善しなければ、次のステップでL1 weight引き上げを検討。**

### 検証手順
1. コード修正 → GitHub push
2. Notebook 03をColabで再実行
3. 確認: 訓練曲線が爆発していないこと、最終rec_loss≈0.1付近、B行列スパーシティ
4. 結果をRecordに記録
5. B行列が依然全結合 → jacobian_l1_weight引き上げ（Step 2へ）
