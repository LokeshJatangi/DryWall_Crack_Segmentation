"""
Shared prompt encoder for FiLM conditioning.

Embeds integer prompt IDs into dense vectors used by FiLM layers
across all model architectures.
"""

import torch
import torch.nn as nn


class PromptEncoder(nn.Module):
    """Embeds prompt_id (int) into a dense vector for FiLM conditioning."""

    def __init__(self, num_prompts: int = 2, embed_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_prompts, embed_dim)

    def forward(self, prompt_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prompt_id: (B,) integer tensor of prompt IDs.

        Returns:
            (B, embed_dim) prompt embeddings.
        """
        return self.embedding(prompt_id)
