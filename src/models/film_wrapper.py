"""
FiLM (Feature-wise Linear Modulation) wrapper for SMP segmentation models.

Applies prompt-conditioned modulation at every encoder scale and injects
an auxiliary prompt signal at the decoder bottleneck.
"""

import torch
import torch.nn as nn

from .prompt_encoder import PromptEncoder


class FiLMBlock(nn.Module):
    """Applies Feature-wise Linear Modulation to a feature map."""

    def __init__(self, embed_dim: int, feature_channels: int):
        super().__init__()
        self.gamma_proj = nn.Linear(embed_dim, feature_channels)
        self.beta_proj = nn.Linear(embed_dim, feature_channels)

        # Initialize to identity modulation: gamma=1, beta=0
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, feature: torch.Tensor, prompt_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature: (B, C, H, W) encoder feature map.
            prompt_embed: (B, embed_dim) prompt embedding.

        Returns:
            Modulated feature map (B, C, H, W).
        """
        gamma = self.gamma_proj(prompt_embed).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta_proj(prompt_embed).unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
        return gamma * feature + beta


class FiLMConditionedModel(nn.Module):
    """
    Wraps an SMP model with FiLM prompt conditioning.

    Architecture:
        Image → SMP Encoder → [features at each scale]
                                    ↓ FiLM modulation per scale
        Prompt ID → PromptEncoder → FiLM γ,β
                                    ↓
                    → SMP Decoder → Segmentation Head → (B,1,H,W)
    """

    def __init__(self, smp_model: nn.Module, prompt_encoder: PromptEncoder, embed_dim: int = 128):
        super().__init__()
        self.encoder = smp_model.encoder
        self.decoder = smp_model.decoder
        self.segmentation_head = smp_model.segmentation_head
        self.prompt_encoder = prompt_encoder

        # FiLM blocks — one per encoder output scale (skip raw input at index 0)
        encoder_channels = list(self.encoder.out_channels)
        self.film_blocks = nn.ModuleList([
            FiLMBlock(embed_dim, ch) for ch in encoder_channels[1:]
        ])

        # Auxiliary decoder injection: project prompt → deepest feature channels
        self.aux_proj = nn.Sequential(
            nn.Linear(embed_dim, encoder_channels[-1]),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor, prompt_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) input images.
            prompt_id: (B,) integer prompt IDs.

        Returns:
            (B, 1, H, W) raw logits (before sigmoid).
        """
        prompt_embed = self.prompt_encoder(prompt_id)  # (B, embed_dim)

        # Encode
        features = self.encoder(image)  # list of feature maps at different scales

        # Apply FiLM to each encoder feature (skip raw input at index 0)
        modulated = [features[0]]
        for feat, film in zip(features[1:], self.film_blocks):
            modulated.append(film(feat, prompt_embed))

        # Auxiliary: add prompt signal to deepest feature
        aux = self.aux_proj(prompt_embed).unsqueeze(-1).unsqueeze(-1)
        modulated[-1] = modulated[-1] + aux.expand_as(modulated[-1])

        # Decode
        decoder_output = self.decoder(modulated)
        mask = self.segmentation_head(decoder_output)
        return mask
