from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class Adapter(nn.Module):
    """(B, n_queries, d_shared) -> (B, n_queries, d_llm). Maps into the LLM's embedding space —
    exactly the shape the embedding layer would have produced for n_queries real tokens.
    """

    def __init__(self, d_shared: int, d_llm: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_shared, d_llm),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_llm, d_llm),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
