# Thinkingモードとhidden state表現に関する先行研究調査

Date: 2026-03-03

## 問い
「ThinkingありのLLMのhidden stateはThinkingなしと質的に異なるか？」

## 結論
**異なる。ただし表現空間は共有しており、活性化パターンと情報構造が質的に異なる。**

---

## 主要論文

### 1. Thinking中に情報密度のピークがある
**"Demystifying Reasoning Dynamics with Mutual Information: Thinking Tokens are Information Peaks in LLM Reasoning"**
- arXiv:2506.02867
- 「Wait」「Therefore」などの転換トークンでhidden stateと正解との相互情報量（MI）が急増する「MI peaks現象」を観測
- 非Thinkingモデルでは同様のピークが弱くかつ少ない
- → Thinkingトークン列の特定位置のhidden stateは情報密度が特別に高い

### 2. ベースモデルの表現空間は共有
**"Base Models Know How to Reason, Thinking Models Learn When"**
- arXiv:2510.07364
- Thinkingモデルが持つ推論能力はベースモデルのhidden stateに既に潜在している
- Thinkingモデルがやっていることは「いつこれらの能力を発動するか」を学習すること
- トークンの12%をステアリングするだけで、Thinkingモデルとの性能差の91%を回復可能
- → 空間は同じ、活性化パターンが異なる

### 3. 3フェーズで特徴分布がシフトする
**"Under the Hood of a Reasoning Model" (Goodfire Research)**
- goodfire.ai/research / github.com/goodfire-ai/r1-interpretability
- DeepSeek R1のSAE分析
- プロンプト処理時・Thinkingトレース中・最終出力生成時の3フェーズで特徴分布が大きくシフト
- 非Thinkingモデルでは見られない構造
- → マルチエージェントのhidden state通信設計に直接影響する重要な知見

### 4. 正誤情報が早期にエンコードされる
**"Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification"**
- arXiv:2504.05419
- 推論モデルのhidden stateに、中間答えの正誤に関する情報が答え形成前から既にエンコードされている
- 2層ニューラルネットのプローブで実証
- → Thinkingモデルのhidden stateは「正しいかどうか」を暗黙的に知っている

### 5. CoT特有のhidden state方向が答えに因果的に寄与
**"How does Chain of Thought Think? Mechanistic Interpretability of Chain-of-Thought Reasoning with Sparse Autoencoding"**
- arXiv:2507.22928
- CoTで活性化するfeatureセットをnoCoTの実行に注入すると答えのlog-probabilityが有意に向上
- → CoT特有のhidden state方向が答えの品質に因果的に寄与

### 6. トークン長は品質の代理指標として不適切
**"Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens"**
- arXiv:2602.13517
- 「Deep-Thinking Ratio（DTR）」を提案：最後のTransformer層で確率分布が安定する前に深い改訂を経たトークンの割合
- 単純なトークン数よりDTRの方が推論品質と強い相関
- → 長く考えれば良いわけではない

### 7. マルチエージェントhidden state通信の有効性
**"Latent Collaboration in Multi-Agent Systems (LatentMAS)"**
- arXiv:2511.20639 / github.com/Gen-Verse/LatentMAS
- 最後層のhidden embeddingで推論し、共有Latent Working Memoryで情報交換
- テキストベースMASと比較して精度+14.6%、トークン使用量70.8〜83.7%削減、速度4〜4.3倍
- ただし**Thinkingあり/なしの比較はまだ行われていない** → 研究の空白

### 8. 連続latent空間での推論
**"COCONUT: Training Large Language Models to Reason in a Continuous Latent Space"**
- arXiv:2412.06769 (Meta / UCSD)
- 最後層のhidden stateを「連続的思考」として直接次ステップの入力embeddingにフィードバック
- 複数の次ステップを同時にエンコードしたBFS的探索が可能
- 論理推論タスクで通常のCoTを上回った

---

## ThoughtCommへの含意

### 論文再現（今回の作業）への影響
- Thinkingオフでも論文の意図（hidden stateによるエージェント間思考通信）は正しく再現できる
- 論文（2025年10月）はThinkingモード登場以前のモデルを想定しており、変数を増やさないためThinkingオフが適切

### 将来の拡張研究として価値がある問い
1. **Thinkingありのhidden stateを通信するとどうなるか？**
   - Goodfireの3相分布シフトがprefix adapterの学習を難しくする可能性
   - 一方、MI peaksのある豊かな表現を通信できれば精度向上の可能性
   - LatentMASはこの比較をまだやっていない → 新規性あり

2. **どのタイミングのhidden stateを通信すべきか？**
   - 最終トークン（現行）vs `</think>`直後（思考完了時点）vs MI peakのタイミング

3. **ThinkingありモデルのB行列（Jacobian構造）はどう変わるか？**
   - ICA/AEで得られるblock-sparse構造がThinking有無で変わるかを比較する研究は新規性が高い

---

## 参考リンク一覧
- https://arxiv.org/abs/2506.02867
- https://arxiv.org/abs/2504.05419
- https://arxiv.org/abs/2510.07364
- https://arxiv.org/abs/2507.22928
- https://aclanthology.org/2025.findings-acl.662/
- https://arxiv.org/abs/2507.10007
- https://www.goodfire.ai/research/under-the-hood-of-a-reasoning-model
- https://arxiv.org/abs/2503.18878
- https://arxiv.org/abs/2509.23676
- https://arxiv.org/abs/2602.20904
- https://arxiv.org/abs/2602.13517
- https://arxiv.org/abs/2502.03373
- https://arxiv.org/abs/2511.20639
- https://arxiv.org/abs/2412.06769
