"""
SegFormer B2 with FiLM prompt conditioning.

Loads pretrained SegFormer, modifies for binary segmentation,
and adds FiLM conditioning at each hierarchical stage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig

from .film_wrapper import FiLMBlock
from .prompt_encoder import PromptEncoder


class SegFormerFiLM(nn.Module):
    """
    SegFormer B2 with FiLM prompt conditioning.

    FiLM modulation is applied to each of the 4 hierarchical stage outputs
    before they enter the decode head.
    """

    # SegFormer B2 stage output channels
    STAGE_CHANNELS = [64, 128, 320, 512]

    def __init__(self, num_prompts: int = 2, embed_dim: int = 128):
        super().__init__()

        # Load pretrained SegFormer B2
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=1,
            ignore_mismatched_sizes=True,
        )

        self.prompt_encoder = PromptEncoder(num_prompts, embed_dim)

        # FiLM blocks for each stage
        self.film_blocks = nn.ModuleList([
            FiLMBlock(embed_dim, ch) for ch in self.STAGE_CHANNELS
        ])

        # Auxiliary injection at the decode head input (deepest stage)
        self.aux_proj = nn.Sequential(
            nn.Linear(embed_dim, self.STAGE_CHANNELS[-1]),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor, prompt_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) input images.
            prompt_id: (B,) integer prompt IDs.

        Returns:
            (B, 1, H, W) raw logits upsampled to input resolution.
        """
        input_h, input_w = image.shape[2], image.shape[3]
        prompt_embed = self.prompt_encoder(prompt_id)  # (B, embed_dim)

        # Get encoder hidden states (4 stages)
        # HuggingFace returns each as (B, seq_len, C) — need (B, C, H, W) for FiLM
        encoder_outputs = self.segformer.segformer(
            pixel_values=image,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = encoder_outputs.hidden_states  # tuple of 4 tensors (B, seq_len, C)

        # Apply FiLM to each stage's hidden state
        # HF may return (B, C, H, W) or (B, seq_len, C) depending on version
        modulated_states = []
        for i, (hs, film) in enumerate(zip(hidden_states, self.film_blocks)):
            if hs.dim() == 3:
                # (B, seq_len, C) — older transformers versions
                B, seq_len, C = hs.shape
                scale = 2 ** (i + 2)
                h = input_h // scale
                w = input_w // scale
                hs_4d = hs.permute(0, 2, 1).reshape(B, C, h, w)
            else:
                # (B, C, H, W) — newer transformers versions
                hs_4d = hs

            modulated_4d = film(hs_4d, prompt_embed)
            modulated_states.append(modulated_4d)

        # Auxiliary injection at deepest stage in (B, C, H, W) format
        B, C, h, w = modulated_states[-1].shape
        aux = self.aux_proj(prompt_embed)  # (B, C)
        modulated_states[-1] = modulated_states[-1] + aux.unsqueeze(-1).unsqueeze(-1)  # broadcast over H, W

        # Pass through decode head (expects list of (B, C, H, W) tensors)
        logits = self.segformer.decode_head(modulated_states)

        # Upsample to input resolution
        logits = F.interpolate(
            logits, size=(input_h, input_w),
            mode='bilinear', align_corners=False,
        )

        return logits
