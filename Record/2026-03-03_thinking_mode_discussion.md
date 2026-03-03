# Qwen3 Thinking Mode に関する考察

Date: 2026-03-03

## 背景

Notebook 02（hidden state収集）実行中に、Qwen3-0.6B が `<think>` タグ付きの内部思考を生成し、
132〜327秒/サンプルという異常な遅さになった。

## 問題の構造

Qwen3のthinkingモードがデフォルトでオンになっており、
512〜2048トークンの大半を `<think>...</think>` の内部思考が占める。
`output_hidden_states=True` と組み合わせると全ステップのhidden statesをメモリに保持するため
さらに遅くなる。

## 採用した対処

`apply_chat_template()` に `enable_thinking=False` を渡してthinkingをオフにした。
非Qwen3モデルへの互換性のため `try/except TypeError` でフォールバックを設ける。

```python
try:
    result = self.tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True,
        enable_thinking=False,
    )
except TypeError:
    result = self.tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True,
    )
```

## 論文再現としての正当性

- ThoughtCommの論文（2025年10月）はthinkingモード登場以前のモデル（LLaMA, Mistral等）を想定
- 論文の「thought」= 生成済みトークン列の**最終層hidden state h_i**（意味表現）
- Qwen3の「thinking」= `<think>`タグ内の**明示的な推論トークン列**（別物）
- thinkingオフでも論文の意図（hidden stateによるエージェント間思考通信）は正しく再現できる

## Thinkingモードの現状

2024年後半〜2025年にかけて主流モデルに急速に普及：

| モデル | Thinking実装 |
|--------|-------------|
| OpenAI o1/o3/o4-mini | Chain-of-Thought |
| Claude 3.7 Sonnet | Extended Thinking |
| Gemini 2.0 Flash Thinking | Thinking mode |
| Qwen3 | `<think>` tokens |
| DeepSeek R1 | `<think>` tokens |

数学・コーディング・論理推論では特にthinkingモードが標準になっている。

## 将来の拡張研究としての可能性

ThoughtCommをthinkingモデルに拡張すると以下の分析が可能：

1. **思考プロセスとhidden stateの対応分析**
   - `<think>`内トークン列と最終hidden stateの関係
   - 長く考えたエージェントのhidden stateは構造が異なるか？

2. **思考共有 vs 回答共有の比較**
   - 通常版: 回答生成後のhidden stateを共有
   - 拡張版: 思考中のhidden stateを共有 → どの段階の思考共有が最も精度に寄与するか？

3. **Jacobian構造の比較**
   - thinkingオン/オフでB行列の構造がどう変わるか
   - 思考モードがエージェント間依存構造を変えるか？

4. **エージェント間の思考収束速度**
   - debateラウンドが進むにつれて思考が収束/発散するかを追跡

これらはThoughtCommの**拡張研究**として独自の価値がある。
まず論文通りの再現を完成させてから検討する。
