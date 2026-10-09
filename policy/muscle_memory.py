"""Muscle Memory: a looped-transformer motor policy for UniBot-V1.

One shared transformer block, iterated `num_loops` times, so depth is a compute
dial rather than a pile of parameters. Token sequence:

    [ACT] [LANG...] [PROPRIO x H]

Outputs an action chunk of shape (action_chunk_size, 16) in normalised units:

    [left_arm(7), right_arm(7), left_gripper(1), right_gripper(1)]

which slices 1:1 onto the challenge's `action.left_arm` / `action.right_arm` /
`action.*_gripper` keys, plus a scalar confidence gate.

About 0.8M parameters, CPU inference around 3 ms per chunk.

The block is inlined here (rather than imported) so this repository is
self-contained: the evaluator only needs `policy/` and `checkpoints/`.
"""

import hashlib

import torch
import torch.nn as nn

STATE_DIM = 16  # 7 left + 7 right joints, 2 grippers


def hash_tokens(text: str, vocab_size: int, max_tokens: int) -> list[int]:
    """Deterministic word-hash tokenisation (no external tokenizer).

    The same string always maps to the same ids on any machine, so the
    checkpoint stays portable to the eval robot.
    """
    words = "".join(c if c.isalnum() else " " for c in text.lower()).split()
    ids = []
    for w in words[:max_tokens]:
        h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(), "big")
        ids.append(h % vocab_size)
    while len(ids) < max_tokens:
        ids.append(0)  # pad id
    return ids


class TransformerBlock(nn.Module):
    """Lightweight pre-norm transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        return self.ln2(x + self.ffn(x))


class MuscleMemoryPolicy(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = STATE_DIM,
                 action_chunk_size: int = 8, proprio_horizon: int = 4,
                 lang_vocab: int = 4096, lang_max_tokens: int = 12,
                 embed_dim: int = 128, num_heads: int = 4, num_loops: int = 4):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.proprio_horizon = proprio_horizon
        self.lang_vocab = lang_vocab
        self.lang_max_tokens = lang_max_tokens

        self.lang_embed = nn.Embedding(lang_vocab + 1, embed_dim, padding_idx=0)
        self.proprio_embed = nn.Linear(state_dim, embed_dim)
        self.proprio_pos = nn.Embedding(proprio_horizon, embed_dim)

        self.act_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.act_query, std=0.02)

        # THE looped trunk: one block reused num_loops times
        self.looped_block = TransformerBlock(embed_dim, num_heads)
        self.loop_embed = nn.Embedding(num_loops, embed_dim)
        self.ln_final = nn.LayerNorm(embed_dim)

        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, action_chunk_size * action_dim),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid(),
        )

    def _tokens(self, lang_ids, proprio):
        B = proprio.shape[0]
        parts = [self.act_query.expand(B, -1, -1)]
        parts.append(self.lang_embed(lang_ids))                    # (B, L, D)
        prop = self.proprio_embed(proprio) + self.proprio_pos.weight
        parts.append(prop)                                         # (B, H, D)
        return torch.cat(parts, dim=1)

    def forward(self, lang_ids, proprio, extra_loops: int = 0):
        x = self._tokens(lang_ids, proprio)
        num_loops = self.loop_embed.num_embeddings + extra_loops
        for i in range(num_loops):
            x = x + self.loop_embed.weight[min(i, self.loop_embed.num_embeddings - 1)]
            x = self.looped_block(x)

        act_token = self.ln_final(x[:, 0])  # [ACT] query token
        chunk = self.action_head(act_token).view(-1, self.action_chunk_size, self.action_dim)
        return chunk, self.confidence_head(act_token)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())