import torch
import torch.nn as nn
import math

VOCAB_SIZE = 256
EMBEDDING_DIM = 128
MAX_SEQ_LEN = 1024
NUM_HEADS = 4
NUM_LAYERS = 4
NUM_EXPERTS = 4


class ByteEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBEDDING_DIM
        )

    def forward(self, tokens):
        return self.token_embedding(tokens)


class RoPE(nn.Module):
    def __init__(self):
        super().__init__()

        head_dim = EMBEDDING_DIM // NUM_HEADS

        position = torch.arange(
            MAX_SEQ_LEN,
            dtype=torch.float32
        )

        frequency = torch.arange(
            0,
            head_dim,
            2,
            dtype=torch.float32
        )

        frequency = 1.0 / (
            10000 ** (frequency / head_dim)
        )

        angles = position[:, None] * frequency[None, :]

        self.register_buffer(
            "cos",
            torch.cos(angles)
        )

        self.register_buffer(
            "sin",
            torch.sin(angles)
        )

    def forward(self, q, k):
        seq_len = q.shape[-2]

        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_even = q[..., 0::2]
        q_odd = q[..., 1::2]

        k_even = k[..., 0::2]
        k_odd = k[..., 1::2]

        q_rotated = torch.stack(
            [
                q_even * cos - q_odd * sin,
                q_even * sin + q_odd * cos
            ],
            dim=-1
        )

        k_rotated = torch.stack(
            [
                k_even * cos - k_odd * sin,
                k_even * sin + k_odd * cos
            ],
            dim=-1
        )

        return (
            q_rotated.flatten(-2),
            k_rotated.flatten(-2)
        )


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_heads = NUM_HEADS
        self.head_dim = EMBEDDING_DIM // NUM_HEADS

        self.qkv = nn.Linear(
            EMBEDDING_DIM,
            EMBEDDING_DIM * 3
        )

        self.output_projection = nn.Linear(
            EMBEDDING_DIM,
            EMBEDDING_DIM
        )

        self.rope = RoPE()

    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        q, k = self.rope(q, k)

        scores = q @ k.transpose(-2, -1)

        scores = scores / math.sqrt(
            self.head_dim
        )

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device
            ),
            diagonal=1
        ).bool()

        scores = scores.masked_fill(
            causal_mask,
            float("-inf")
        )

        weights = torch.softmax(
            scores,
            dim=-1
        )

        output = weights @ v

        output = output.transpose(
            1,
            2
        ).contiguous()

        output = output.view(
            batch_size,
            seq_len,
            embedding_dim
        )

        return self.output_projection(output)


class LinearAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.query = nn.Linear(
            EMBEDDING_DIM,
            EMBEDDING_DIM
        )

        self.key = nn.Linear(
            EMBEDDING_DIM,
            EMBEDDING_DIM
        )

        self.value = nn.Linear(
            EMBEDDING_DIM,
            EMBEDDING_DIM
        )

        self.output_projection = nn.Linear(
            EMBEDDING_DIM,
            EMBEDDING_DIM
        )

    def forward(self, x):
        q = torch.sigmoid(
            self.query(x)
        )

        k = torch.sigmoid(
            self.key(x)
        )

        v = self.value(x)

        batch_size, seq_len, dim = x.shape

        state = torch.zeros(
            batch_size,
            dim,
            dim,
            device=x.device,
            dtype=x.dtype
        )

        normalizer = torch.zeros(
            batch_size,
            dim,
            device=x.device,
            dtype=x.dtype
        )

        outputs = []

        for t in range(seq_len):

            kt = k[:, t]
            vt = v[:, t]
            qt = q[:, t]

            state = state + (
                kt.unsqueeze(-1) *
                vt.unsqueeze(-2)
            )

            normalizer = normalizer + kt

            numerator = torch.bmm(
                qt.unsqueeze(1),
                state
            ).squeeze(1)

            denominator = (
                torch.sum(
                    qt * normalizer,
                    dim=-1,
                    keepdim=True
                ) + 1e-6
            )

            output = numerator / denominator

            outputs.append(output)

        output = torch.stack(
            outputs,
            dim=1
        )

        return self.output_projection(output)


class Expert(nn.Module):
    def __init__(self):
        super().__init__()

        hidden_dim = EMBEDDING_DIM * 4

        self.network = nn.Sequential(
            nn.Linear(
                EMBEDDING_DIM,
                hidden_dim
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                EMBEDDING_DIM
            )
        )

    def forward(self, x):
        return self.network(x)


class MoE(nn.Module):
    def __init__(self):
        super().__init__()

        self.router = nn.Linear(
            EMBEDDING_DIM,
            NUM_EXPERTS
        )

        self.experts = nn.ModuleList([
            Expert()
            for _ in range(NUM_EXPERTS)
        ])

    def forward(self, x):
        router_logits = self.router(x)

        router_probs = torch.softmax(
            router_logits,
            dim=-1
        )

        top_expert = torch.argmax(
            router_probs,
            dim=-1
        )

        output = torch.zeros_like(x)

        for expert_index, expert in enumerate(
            self.experts
        ):
            mask = (
                top_expert == expert_index
            )

            if mask.any():
                selected_tokens = x[mask]

                expert_output = expert(
                    selected_tokens
                )

                weights = router_probs[
                    mask,
                    expert_index
                ].unsqueeze(-1)

                output[mask] = (
                    expert_output * weights
                )

        return output


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention_norm = nn.LayerNorm(
            EMBEDDING_DIM
        )

        self.full_attention = SelfAttention()

        self.linear_attention = LinearAttention()

        self.mlp_norm = nn.LayerNorm(
            EMBEDDING_DIM
        )

        self.moe = MoE()

    def forward(self, x):
        normalized = self.attention_norm(x)

        full_output = self.full_attention(
            normalized
        )

        linear_output = self.linear_attention(
            normalized
        )

        x = x + full_output + linear_output

        x = x + self.moe(
            self.mlp_norm(x)
        )

        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = ByteEmbedding()

        self.blocks = nn.ModuleList([
            TransformerBlock()
            for _ in range(NUM_LAYERS)
        ])

        self.final_norm = nn.LayerNorm(
            EMBEDDING_DIM
        )

    def forward(self, tokens):
        x = self.embedding(tokens)

        for block in self.blocks:
            x = block(x)

        return self.final_norm(x)


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.transformer = Transformer()

        self.output_bias = nn.Parameter(
            torch.zeros(VOCAB_SIZE)
        )

    def forward(self, tokens):
        x = self.transformer(tokens)

        embedding_weights = (
            self.transformer
            .embedding
            .token_embedding
            .weight
        )

        logits = x @ embedding_weights.T

        return logits + self.output_bias


if __name__ == "__main__":
    model = LanguageModel()

    tokens = torch.tensor([
        [72, 101, 108, 108, 111]
    ])

    logits = model(tokens)

    print("Input shape:", tokens.shape)
    print("Logits shape:", logits.shape)
    print("Vocabulary size:", VOCAB_SIZE)
    print("Number of experts:", NUM_EXPERTS)
    print("Hybrid attention: enabled")