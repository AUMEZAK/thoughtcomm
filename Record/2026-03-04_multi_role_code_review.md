# 多角的コードレビュー結果 (2026-03-04)

## 背景
Notebook 05の第1回フルパイプライン実行結果に対し、問題・課題を洗い出すため、
複数の専門家ロールでコード全体を精読した。

## 発端: Single Answer MATH 5% vs Debate 51%
10倍の精度差は文献上前例がなく、バグの可能性が高いと判断。

---

## ロール別レビュー結果

### 👨‍🔬 Role 1: 論文再現の研究者

**問題A: Hidden Stateの収集タイミング**
- `hidden_state_collector.py`: 全ラウンド(Round 0 + Round 1)のhidden stateを収集
- H_all = (100問 × 2ラウンド, 3072) = (200, 3072)
- Round 0（独立回答）とRound 1（議論後）が混在 → AE訓練データの分布が不均一

**問題B: Adapter訓練のL_commが論文Eq.12と不一致**
- `train_adapter.py` 165行: `gen_ids = logits[...].argmax(dim=-1)`
- argmaxは微分不可能 → cosine_lossの勾配がadapterパラメータに流れない
- 実質LM lossのみで訓練されており、論文のL_commの半分が機能していない

**問題C: prefix_length=1の妥当性**
- 論文Fig 5ではm=1,4,8,16をアブレーション
- 現在m=1で固定。情報伝達量が不足している可能性

### 👨‍💻 Role 2: コーディングエンジニア

**問題D（確定バグ）: `enable_thinking=False`の不一致**
- `debate.py` 107行: `enable_thinking=False` あり
- `thought_comm.py` 141行 (single answer): なし
- `train_adapter.py` 127行: なし
- 3箇所中1箇所しか設定されていない

**問題E: `_generate_with_prefix`のhidden state再計算が入力欠落**
- `debate.py` 192-193行: `torch.cat([prefix, gen_embeds], dim=1)`
- `token_embeds`（入力プロンプト）が含まれていない
- 正しくは `torch.cat([prefix, token_embeds, gen_embeds], dim=1)`
- 現在num_rounds=2なので影響なし（round 1のhidden stateは使われない）
- num_rounds>2では壊れる

**問題F: `_generate_with_prefix`のdecodeがモデル/HF依存**
- `generate(inputs_embeds=...)`の返り値がモデル依存
- 生成トークンのみか、入力位置のダミートークンを含むか不定

### 👨‍🔧 Role 3: OSS LLM専門家（Qwen3特化） ← **最重要**

**問題G: Qwen3 thinking modeの実際の挙動（問題Dの因果説明）**
- `enable_thinking`未指定時、Qwen3はthinking ONがデフォルト
- thinking ON: `<think>長い推論...</think>\n\n回答` を生成
- 推論部分が`max_new_tokens=512`の大部分を消費
- `\boxed{}`に到達しない → `extract_boxed_answer`がNoneを返す → 不正解
- **これがMATH Single Answer 5%の直接原因**
- Debateは`enable_thinking=False`なので直接回答 → 512トークン内に`\boxed{}`が収まる → 51%

**このロールが最もクリティカル。** コードの差分（問題D）は誰でも見つけられるが、
「Qwen3のthinking modeが512トークンの大半を消費し、\boxed{}に到達しない」
という因果関係を繋げられたのはQwen3の挙動を知る専門家のみ。

**問題H: `pad_token = eos_token`の影響**
- 生成の早期終了に影響する可能性

### 📊 Role 4: 評価方法論の専門家

**問題I: Consensus計算がNoneをスキップ**
- `metrics.py` 54行: `\boxed{}`が見つからないとその問題をスキップ
- thinking ONでNone率が高ければ分母が大幅減少

**問題J: MATH ground truthのフォールバック**
- `_extract_boxed`で`\boxed{}`がなければ最終行を返す → grading不正の可能性

### 📈 Role 5: 統計・データパイプライン専門家

**問題K: AE訓練データにRound 0/1が混在**
- 分布の異なるhidden stateが混在 → AEが中途半端な潜在空間を学習

**問題L: Adapter過学習リスク**
- パラメータ131万 vs 訓練データ100件

---

## 問題の優先度

| 優先 | 問題 | 種類 | 影響 |
|---|---|---|---|
| **1** | D+G: `enable_thinking=False`不一致 | 確定バグ | Single Answer精度が根本的に不正 |
| **2** | B: cosine_lossの勾配が流れない | 設計不良 | Adapter訓練が不完全 |
| **3** | E: prefix生成時のhidden state欠落 | 潜在バグ | num_rounds>2で発症 |
| **4** | F: prefix生成時のdecode不定 | 潜在バグ | モデル/HF依存 |
| **5** | K: AE訓練データのラウンド混在 | 設計疑問 | AEの品質に影響 |
| **6** | A: hidden state収集タイミング | 未検証 | 論文との乖離 |

## 次のアクション
1. 問題D: `enable_thinking=False`を3箇所に統一（1行修正×2箇所）
2. Notebook 05を再実行してSingle Answerの精度変化を確認
3. 問題B: cosine_lossの勾配問題を修正（adapter訓練ロジック書き直し）
4. 必要に応じてNotebook 04→05を再実行
