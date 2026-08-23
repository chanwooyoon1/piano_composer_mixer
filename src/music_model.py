import math

import torch
import torch.nn as nn
import torch.nn.functional as F


D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 8
FFN_DIM = 2048
DROPOUT = 0.10
NUM_COMPOSERS = 6
PAD_TOKEN = 0


def apply_rope(x):
    """
    x: (batch, heads, time, head_dim)
    """
    _, _, length, head_dim = x.shape
    device = x.device

    positions = torch.arange(
        length,
        device=device,
        dtype=torch.float32,
    )

    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, head_dim, 2, device=device).float()
        / head_dim
    )

    angles = positions[:, None] * frequencies[None, :]
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    return torch.stack(
        (rotated_even, rotated_odd),
        dim=-1,
    ).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch, length, channels = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        q = apply_rope(q)
        k = apply_rope(k)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        x = x.transpose(1, 2).contiguous().view(
            batch, length, channels
        )

        return self.out(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model,
            num_heads,
            dropout,
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ComposerMusicTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_composers,
        d_model,
        num_heads,
        num_layers,
        ffn_dim,
        dropout,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=PAD_TOKEN,
        )

        self.composer_embedding = nn.Embedding(
            num_composers,
            d_model,
        )

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                ffn_dim,
                dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens, composer_ids):
        x = self.token_embedding(tokens)

        composer = self.composer_embedding(composer_ids)[:, None, :]
        x = self.dropout(x + composer)

        for block in self.blocks:
            x = block(x)

        return self.lm_head(self.norm(x))

def load_music_model(checkpoint_path):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)
    print("Loading checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = ComposerMusicTransformer(
        vocab_size=checkpoint["vocab_size"],
        num_composers=checkpoint["num_composers"],
        d_model=checkpoint["d_model"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        ffn_dim=checkpoint["ffn_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.lm_head.weight = model.token_embedding.weight

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, device
