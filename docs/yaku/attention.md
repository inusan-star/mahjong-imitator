# 麻雀構造特化型Transformer (Mahjong Structure-Aware Transformer) の解説

このモデルは、汎用的なTransformerアーキテクチャに、麻雀の物理的なルール（牌の接続関係や属性）を**「トポロジー・マスク（幾何学的制約）」**として組み込んだ、極めて独創的なニューラルネットワークです。

---

## 1. 全体コンセプト：Inductive Bias（構造的バイアス）の注入

通常のAIは「1萬」と「2萬」が隣同士であることを数百万回の対局から学習しなければなりませんが、本モデルは**「牌の並び（順子）」「同じ牌（刻子）」「端の牌（チャンタ・タンヤオ）」**といった麻雀の基本トポロジーを数式として最初から持っています。これにより、極めて少ないデータで高度な「役の形」を理解することが可能になります。

---

## 2. コード詳細解説

### 2.1 麻雀トポロジーの定義 (`get_mahjong_topology_masks`)

ここでは、牌同士の「論理的な距離」を計算し、アテンションの初期知識となる行列を生成します。

```python
def get_mahjong_topology_masks(num_tokens: int) -> torch.Tensor:
    # 8つのアテンションヘッドごとに異なる「専門知識（マスク）」を生成
    masks = torch.zeros(config.NUM_HEADS, num_tokens, num_tokens)

    # 1・9・字牌を「境界牌（Boundary）」として定義
    terminals = {0, 8, 9, 17, 18, 26}
    honors = {27, 28, 29, 30, 31, 32, 33}

    for i in range(num_tokens):
        for j in range(num_tokens):
            # --- 4. グローバル・アテンション (Heads 6, 7) ---
            # 35番目のトークン（場況情報）は、すべての牌の情報を等しく参照する
            if i == 34 or j == 34:
                masks[6:8, i, j] = 1.0
                continue

            # --- 1. シーケンス・アテンション (Heads 0, 1) ---
            # 同じ色（萬子・筒子・索子）の中で、距離が近い牌同士を凝視する
            if i < 27 and j < 27 and (i // 9 == j // 9):
                dist = abs(i - j)
                if dist == 1: masks[0:2, i, j] = 2.0  # 隣(2・3など)は非常に重要
                elif dist == 2: masks[0:2, i, j] = 1.0 # 1つ飛ばし(カンチャン)も考慮

            # --- 2. アイデンティカル・アテンション (Heads 2, 3) ---
            # 「同じ牌」や「別スートの同じ数字（三色）」を凝視する
            if i == j:
                masks[2:4, i, j] = 3.0  # 同一牌（対子・刻子）は最も強い繋がり
            elif i < 27 and j < 27 and i % 9 == j % 9:
                masks[2:4, i, j] = 1.5  # 三色同順・同刻の形を捕捉

            # --- 3. バウンダリー・アテンション (Heads 4, 5) ---
            # 1・9・字牌の集まりを監視し、チャンタ、タンヤオ、国士の「範囲」を捉える
            if i in terminals and j in terminals:
                masks[4:6, i, j] = 1.5  # 老頭牌同士
            elif i in honors and j in honors:
                masks[4:6, i, j] = 2.0  # 字牌同士
```

### 2.2 特化型マルチヘッドアテンション (`MultiHeadAttention`)

生成したマスクを実際のアテンション計算に注入します。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, num_tokens: int):
        # ...レイヤー定義...
        # 登録したトポロジー・マスクをモデル内に保存
        self.register_buffer("topology_masks", get_mahjong_topology_masks(num_tokens))
        # 各ヘッドが「どれだけトポロジーを重視するか」を学習するためのパラメータ
        self.topology_scale = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ...Q, K, Vの生成...
        # 標準的なアテンションスコア (Q * K^T)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dimension**0.5)

        # 【核心部】計算されたスコアに、麻雀トポロジーの「偏見」を加算
        # これにより、モデルは最初から「順子」や「刻子」に注目した状態で演算を始める
        attention_scores = attention_scores + (self.topology_scale * self.topology_masks)

        # ソフトマックスで確率変換し、価値(V)を抽出
        attention_weights = self.dropout(F.softmax(attention_scores, dim=-1))
        # ...

```

### 2.3 モデル本体 (`Transformer`)

全体の流れを統括します。

```python
class Transformer(nn.Module):
    def _init_weights(self, module: nn.Module):
        # レイヤーの種類に応じた高度な初期化
        if isinstance(module, nn.Linear):
            # アテンション層や出力層には Xavier初期化（勾配の安定化）
            if hasattr(module, "is_attention_query") or module.out_features == config.OUTPUT_DIM:
                nn.init.xavier_uniform_(module.weight)
            # それ以外（FFNなど）には Kaiming初期化（ReLU用）
            else:
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 牌の情報を高次元ベクトル（Embedding）に変換
        x = self.embedding_layer(x)
        # 2. 位置エンコーディングを加算
        x = self.dropout(x + self.positional_encoding)
        # 3. 複数のTransformerBlock（トポロジーアテンション搭載）を通過
        for block in self.blocks:
            x = block(x)
        # 4. 全情報を1次元に潰して、最終的な「役の確率」を出力
        x = x.reshape(batch_size, -1)
        return self.output_layer(x)

```

---

## 3. このモデルの学術的価値

1. **説明可能性 (Explainability)**:
   `topology_scale` を解析することで、「このAIは平和を予測する際に、確かに順子担当ヘッド（Head 0/1）を重視している」という根拠を提示できます。
2. **パラメータ効率 (Efficiency)**:
   闇雲に層を深くするのではなく、麻雀のドメイン知識を制約として与えることで、軽量なモデルでも「二盃口」や「三色」などの複雑な形を正確に捉えられます。
3. **汎用性の否定**:
   「何でも解ける普通のTransformer」ではなく、「麻雀を解くために最適化された幾何学的Transformer」である点が、専門家にとっても説得力のあるアプローチとなります。
