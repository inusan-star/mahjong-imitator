import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

import src.yaku.exp5.config as config


def get_mahjong_topology_masks(num_tokens: int) -> torch.Tensor:
    """
    Generate Mahjong topology masks with Soft Priors.
    1. Sequence (Heads 0,1): 距離1は強く(2.0)、距離2は中程度(1.0)
    2. Identical (Heads 2,3): 同一牌は最強(3.0)、三色は中程度(1.5)
    3. Boundary (Heads 4,5): 1-9同士、字牌同士、1-9と字牌を分離して重み付け
    4. Global (Heads 6,7): 場況トークンとの接続
    """
    masks = torch.zeros(config.NUM_HEADS, num_tokens, num_tokens)

    terminals = {0, 8, 9, 17, 18, 26}
    honors = {27, 28, 29, 30, 31, 32, 33}

    for i in range(num_tokens):
        for j in range(num_tokens):
            # 4. Global-Attention: 場況(34)は全牌と緩やかに繋ぐ
            if i == 34 or j == 34:
                masks[6:8, i, j] = 1.0
                continue

            # 1. Sequence-Attention (順子構造)
            if i < 27 and j < 27 and (i // 9 == j // 9):
                dist = abs(i - j)
                if dist == 1:
                    masks[0:2, i, j] = 2.0  # 隣接
                elif dist == 2:
                    masks[0:2, i, j] = 1.0  # カンチャン

            # 2. Identical-Attention (刻子・三色構造)
            if i == j:
                masks[2:4, i, j] = 3.0  # 同一牌 (最強)
            elif i < 27 and j < 27 and i % 9 == j % 9:
                masks[2:4, i, j] = 1.5  # 三色 (他色の同数)

            # 3. Boundary-Attention (端牌・字牌構造)
            # 国士やチャンタを考慮し、1-9同士と字牌同士の結合を分ける
            if i in terminals and j in terminals:
                masks[4:6, i, j] = 1.5  # 老頭牌の繋がり
            elif i in honors and j in honors:
                masks[4:6, i, j] = 2.0  # 字牌同士の繋がり
            elif (i in terminals and j in honors) or (i in honors and j in terminals):
                masks[4:6, i, j] = 0.5  # 端と字の緩やかな繋がり

    return masks


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with specialized Mahjong topology kernels.
    """

    def __init__(self, embedding_dim: int, num_heads: int, num_tokens: int):
        super().__init__()
        assert embedding_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dimension = embedding_dim // num_heads

        self.query_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(config.DROPOUT)

        # 牌譜のトポロジーを学習可能なスケールで注入
        self.register_buffer("topology_masks", get_mahjong_topology_masks(num_tokens))
        self.topology_scale = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.is_attention_query = True
        self.is_attention_key = True
        self.is_attention_value = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = x.shape

        queries = (
            self.query_layer(x)
            .reshape(batch_size, sequence_length, self.num_heads, self.head_dimension)
            .transpose(1, 2)
        )
        keys = (
            self.key_layer(x).reshape(batch_size, sequence_length, self.num_heads, self.head_dimension).transpose(1, 2)
        )
        values = (
            self.value_layer(x)
            .reshape(batch_size, sequence_length, self.num_heads, self.head_dimension)
            .transpose(1, 2)
        )

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dimension**0.5)

        attention_scores = attention_scores + (self.topology_scale * self.topology_masks)

        attention_weights = self.dropout(F.softmax(attention_scores, dim=-1))
        attention_output = (
            torch.matmul(attention_weights, values).transpose(1, 2).reshape(batch_size, sequence_length, embedding_dim)
        )

        return self.output_layer(attention_output)


class TransformerBlock(nn.Module):
    """
    Transformer block with topology-aware attention.
    """

    def __init__(self, embedding_dim: int, num_heads: int, num_tokens: int, feed_forward_dimension: int):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads, num_tokens)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(config.DROPOUT)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, feed_forward_dimension),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(feed_forward_dimension, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class Transformer(nn.Module):
    """
    Mahjong Structure-Aware Transformer.
    """

    def __init__(
        self,
        *,
        input_dimension: int,
        num_tokens: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        feed_forward_dimension: int,
        output_dim: int,
    ):
        super().__init__()

        self.input_dimension = input_dimension
        self.num_tokens = num_tokens

        self.embedding_layer = nn.Linear(input_dimension, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_tokens, embedding_dim))
        self.dropout = nn.Dropout(config.DROPOUT)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embedding_dim, num_heads, num_tokens, feed_forward_dimension) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim * num_tokens, output_dim)

        self.apply(self._init_weights)
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            if hasattr(module, "is_attention_query") or module.out_features == config.OUTPUT_DIM:
                nn.init.xavier_uniform_(module.weight)
            else:
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, input_dimension = x.shape

        assert (
            input_dimension == self.input_dimension
        ), f"Actual input dim ({input_dimension}) must be equal to expected input dim ({self.input_dimension})"
        assert (
            num_tokens == self.num_tokens
        ), f"Actual token count ({num_tokens}) must be equal to expected token count ({self.num_tokens})"

        x = self.embedding_layer(x)
        x = self.dropout(x + self.positional_encoding)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.reshape(batch_size, -1)
        x = self.output_layer(x)

        return x


if __name__ == "__main__":
    model = Transformer(
        input_dimension=config.INPUT_DIM,
        num_tokens=config.NUM_TOKENS,
        embedding_dim=config.EMBEDDING_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        feed_forward_dimension=config.HIDDEN_DIM,
        output_dim=config.OUTPUT_DIM,
    )

    summary(model, input_size=(config.LEARNING_BATCH_SIZE, config.NUM_TOKENS, config.INPUT_DIM))
