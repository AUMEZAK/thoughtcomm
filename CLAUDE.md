# ThoughtComm - Session Handoff

## Project Overview
NeurIPS 2025 Spotlight 論文 "Thought Communication in Multiagent Collaboration" (arXiv:2510.20733) の非公式再現。
LLMエージェント間で自然言語ではなく「潜在的思考（latent thoughts）」を直接共有する手法。

- GitHub: https://github.com/AUMEZAK/thoughtcomm
- 論文: https://arxiv.org/abs/2510.20733
- ブランチ: `claude/investigate-codebase-z2oid`（修正作業中）

## 背景情報（ユーザーの過去セッションより）
- 2026-02-15 の会話ログ・作業記録がローカルに保存済み
  - `D:\Takaede2025\2026-02-15_Thought_Communication\Record\`
- PostgreSQL (pgvector) に 27 チャンク登録済み
- MEMORY.md: `C:\Users\E9T\.claude\projects\D--Takaede2025-2026-02-15-Thought-Communication\memory\MEMORY.md`

## 今回のセッションで行ったこと

### 1. リポジトリ構造の調査
- 5つの Colab ノートブック（順番に実行する設計）
- `src/`: configs, data, models, pipeline, training, evaluation, utils
- Qwen-3-0.6B と Phi-4-mini をサポート

### 2. Notebook 1 の速度改善（commit: cc029cb）
- `stochastic_jacobian_l1` を `vmap(jacrev(...))` でベクトル化（10-30x高速化）
- フォールバックとしてループ版を保持
- ファイル: `training/jacobian_utils.py`

### 3. Notebook 1 の実行結果分析（Colab A100 で実行済み）

**問題: MCC が極端に低い**

実行結果:
```
R² マトリックス (Ours):
[[0.206, 0.337, 0.194],
 [0.205, 0.323, 0.202],
 [0.227, 0.351, 0.195]]

R² マトリックス (Baseline):  ← Ours とほぼ同一！
[[0.208, 0.330, 0.195],
 [0.209, 0.332, 0.196],
 [0.221, 0.365, 0.197]]

MCC sweep:
  DIM=128: 0.2129  (期待値: >0.75)
  DIM=256: 0.1626
  DIM=512: 0.1268
  DIM=1024: 未完了
```

期待される結果:
- R² (Ours): 対角 0.6-0.8、非対角 0-0.1
- R² (Baseline): 分離されていない（非対角も高い）
- MCC: 全次元で >0.75

### 4. 原因分析

**根本原因の仮説: `jacobian_l1_weight` がこの次元に対して不十分**

トレーニングログ:
```
Epoch 200: rec=0.003457, jac=0.1043
ペナルティ寄与: 0.01 × 0.1043 = 0.001043 (全体の23%のみ)
```

L1推定のスケーリング: `mean(|J|) × (n_h / sample_rows)`
- n_h=128, k=32 → ×4
- n_h=3072, k=64 → ×48（本番実験）
- 合成実験では実効ペナルティが約12倍弱い

### 5. 仮修正（commit: 2f4f830）
- Cell 5: `jacobian_l1_weight` を 0.01 → 0.1 に変更
- Cell 11 (MCC sweep): 次元に応じた adaptive lambda スケーリングを追加
- **未検証** — 論文の正確なハイパーパラメータを確認する必要あり

## 未解決の課題（最優先）

### 1. 論文の確認が必要
- `/learn https://arxiv.org/html/2510.20733` で論文を読み込む
- Section 5.1 の合成実験のハイパーパラメータ（特に lambda）を確認
- Appendix の詳細設定を確認

### 2. lambda 修正の検証
- lambda=0.1 で Notebook 1 を再実行し、MCC >0.75 になるか確認
- ならない場合、他の原因を調査:
  - 勾配フロー（`create_graph=True` が decoder params に正しく伝播しているか）
  - AE アーキテクチャ（hidden_dim=256 で十分か）
  - データ正規化（X のスケール）

### 3. Notebook 2-5 はまだ未実行

## ファイル構成（重要なもの）

```
training/jacobian_utils.py  ← vmap版に書き換え済み（ブランチ上）
notebooks/01_synthetic_experiment.ipynb  ← lambda修正済み（ブランチ上）
configs/config.py  ← デフォルト lambda=0.01（変更なし）
training/train_autoencoder.py  ← 変更なし
models/autoencoder.py  ← 変更なし
```

## ブランチ状態
- `main`: Colab 実行結果（低MCC）が保存されている
- `claude/investigate-codebase-z2oid`: 速度改善 + lambda仮修正
