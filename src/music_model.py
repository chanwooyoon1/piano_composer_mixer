"""
Defines the composer conditioned Transformer used for structured piano music
generation. The model uses rotary position information, causal self attention,
composer embeddings, residual Transformer blocks, and shared token weights.
It also provides a function for loading a trained checkpoint on CUDA, MPS, or CPU.
"""

# Import mathematical functions and PyTorch model components.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the architecture dimensions used during training.
D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 8
FFN_DIM = 2048
DROPOUT = 0.10
NUM_COMPOSERS = 6
PAD_TOKEN = 0


# Apply rotary position information to attention vectors.
def apply_rope(x):
    """
    x: (batch, heads, time, head_dim)
    """
    _, _, length, head_dim = x.shape
    device = x.device

    # Create one position index for every token in the sequence.
    positions = torch.arange(
        length,
        device=device,
        dtype=torch.float32,
    )

    # Create rotation frequencies for pairs of attention dimensions.
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, head_dim, 2, device=device).float()
        / head_dim
    )

    # Convert positions and frequencies into rotation angles.
    angles = positions[:, None] * frequencies[None, :]
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    # Separate even and odd dimensions before rotation.
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Rotate each pair of dimensions.
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    # Combine the rotated dimensions into their original layout.
    return torch.stack(
        (rotated_even, rotated_odd),
        dim=-1,
    ).flatten(-2)


# Implement causal attention for autoregressive token prediction.
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()

        # Ensure that every attention head receives the same dimension.
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        # Produce query, key, and value tensors with one projection.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch, length, channels = x.shape

        # Split the combined projection into query, key, and value tensors.
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Separate query vectors into multiple attention heads.
        q = q.view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Separate key vectors into multiple attention heads.
        k = k.view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Separate value vectors into multiple attention heads.
        v = v.view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Add rotary position information to queries and keys.
        q = apply_rope(q)
        k = apply_rope(k)

        # Prevent each sequence position from accessing future tokens.
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Merge all attention heads back into the model dimension.
        x = x.transpose(1, 2).contiguous().view(
            batch, length, channels
        )

        return self.out(x)


# Combine causal attention and a feed forward network.
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout):
        super().__init__()

        # Normalize inputs before causal attention.
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model,
            num_heads,
            dropout,
        )

        # Normalize attention outputs before the feed forward network.
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Apply attention and the feed forward network with residual connections.
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# Predict music tokens while conditioning on a composer identity.
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

        # Convert discrete music tokens into continuous vectors.
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=PAD_TOKEN,
        )

        # Learn one conditioning vector for each composer category.
        self.composer_embedding = nn.Embedding(
            num_composers,
            d_model,
        )

        self.dropout = nn.Dropout(dropout)

        # Stack the configured number of Transformer blocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                ffn_dim,
                dropout,
            )
            for _ in range(num_layers)
        ])

        # Normalize final representations and predict vocabulary scores.
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens, composer_ids):
        # Embed the input music tokens.
        x = self.token_embedding(tokens)

        # Add the selected composer representation to every token.
        composer = self.composer_embedding(composer_ids)[:, None, :]
        x = self.dropout(x + composer)

        # Process the sequence through every Transformer block.
        for block in self.blocks:
            x = block(x)

        # Return one vocabulary prediction for every sequence position.
        return self.lm_head(self.norm(x))


# Load a trained model on the best available execution device.
def load_music_model(checkpoint_path):
    # Prefer CUDA, then Apple MPS, and finally CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)
    print("Loading checkpoint...")

    # Load model parameters and architecture information.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # Reconstruct the exact architecture stored in the checkpoint.
    model = ComposerMusicTransformer(
        vocab_size=checkpoint["vocab_size"],
        num_composers=checkpoint["num_composers"],
        d_model=checkpoint["d_model"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        ffn_dim=checkpoint["ffn_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)

    # Share the input token embedding with the output projection.
    model.lm_head.weight = model.token_embedding.weight

    # Restore trained parameters and enable evaluation behavior.
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, device
