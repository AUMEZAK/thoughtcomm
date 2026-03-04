# 第1回フルパイプライン実行結果 (2026-03-04)

## 概要
- モデル: Qwen/Qwen3-0.6B (hidden=1024, n_h=3072)
- エージェント数: 3（同一モデル、temperature=0.7）
- Debateラウンド数: 2
- 評価データ: MATH Level 3 × 100, GSM8K × 100
- 実行環境: Google Colab (H100)

## Table 1 再現結果

| Method | MATH Acc (%) | MATH Cons (%) | GSM8K Acc (%) | GSM8K Cons (%) |
|---|---|---|---|---|
| Single Answer | 5.0 | — | 48.0 | — |
| Debate Only | 51.0 | 65.0 | 67.7 | 72.0 |
| ThoughtComm (Ours) | 52.3 | 62.0 | 63.3 | 74.0 |

### Agent別精度

**Debate Only:**
| Agent | MATH Acc | GSM8K Acc |
|---|---|---|
| Agent 0 | 50.0% | 67.0% |
| Agent 1 | 52.0% | 68.0% |
| Agent 2 | 51.0% | 68.0% |

**ThoughtComm:**
| Agent | MATH Acc | GSM8K Acc |
|---|---|---|
| Agent 0 | 51.0% | 64.0% |
| Agent 1 | 55.0% | 63.0% |
| Agent 2 | 51.0% | 63.0% |

## パイプライン各ステップの状態

### Notebook 02: Hidden State収集
- MATH: (200, 3072), GSM8K: (200, 3072)
- num_train=100, num_eval=100

### Notebook 03: AE訓練（3回実行）
- 1回目: 訓練爆発 (rec_loss=13807)
- 2回目: L1安定化 (rec_loss=0.0125, スパーシティ89.9%, alpha全て3)
- 3回目: Group Lasso (rec_loss=0.0135, スパーシティ99.6%)

**B行列 alpha分布（最終）:**
| alpha | 次元数 | 割合 |
|---|---|---|
| alpha=0 | 26 | 2.5% |
| alpha=1 | 98 | 9.6% |
| alpha=2 | 263 | 25.7% |
| alpha=3 | 637 | 62.2% |

### Notebook 04: Adapter訓練
- パラメータ数: 1,312,768
- 最終loss: 0.054
- Agreement weights: [1.0, 1.389, 1.238, 1.036]

### Notebook 05: 評価
- n=100で実行（約3.5時間）

## 結果の評価

### ThoughtCommの効果
- **MATH**: Debate比 +1.3%（統計的に有意ではない、SE≈±5%）
- **GSM8K**: Debate比 -4.3%（悪化、ただし有意ではない）
- **Consensus**: MATH悪化(65→62%)、GSM8K微増(72→74%)

### 結論
ThoughtCommの効果はこの構成では確認できなかった。

### 考えられる原因
1. alpha分布の62.2%がalpha=3で、agreement reweightingの差別化が弱い
2. 同一モデル3エージェントでは隠れ状態の差が小さすぎる
3. Prefix Adapterの訓練データ(100件)が少なすぎる可能性
4. AE訓練データ(200件)が3072次元に対して不足

## コミット履歴
- `b4c37ba` — Notebook 02 結果
- `16976a4` — Notebook 03 1回目（爆発）
- `6908179` — AE安定化修正
- `d8bb718` — Notebook 03 2回目（L1安定化）
- `bf23c56` — Group Lasso実装
- `c44e8cc` — Notebook 03 3回目（Group Lasso）
- `b31ef68` — Notebook 04 結果
- `db65362` — eval数n=100に修正
- `f656084` — Notebook 05 結果
