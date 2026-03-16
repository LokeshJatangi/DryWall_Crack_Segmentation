"""
Factory for creating SMP-based segmentation models with FiLM conditioning.

Supports: Unet, UnetPlusPlus, DeepLabV3Plus.
"""

import segmentation_models_pytorch as smp

from .film_wrapper import FiLMConditionedModel
from .prompt_encoder import PromptEncoder


SUPPORTED_ARCHS = {
    'Unet': smp.Unet,
    'UnetPlusPlus': smp.UnetPlusPlus,
    'DeepLabV3Plus': smp.DeepLabV3Plus,
}


def create_smp_model(
    arch: str = "Unet",
    encoder_name: str = "resnet34",
    num_prompts: int = 2,
    embed_dim: int = 128,
) -> FiLMConditionedModel:
    """
    Create an SMP model wrapped with FiLM prompt conditioning.

    Args:
        arch: Architecture name (Unet, UnetPlusPlus, DeepLabV3Plus).
        encoder_name: Backbone encoder (e.g., resnet34, resnet50, efficientnet-b3).
        num_prompts: Number of distinct text prompts.
        embed_dim: Dimension of prompt embeddings.

    Returns:
        FiLMConditionedModel with the specified architecture.
    """
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unknown arch '{arch}'. Supported: {list(SUPPORTED_ARCHS.keys())}")

    smp_model = SUPPORTED_ARCHS[arch](
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )

    prompt_enc = PromptEncoder(num_prompts, embed_dim)
    return FiLMConditionedModel(smp_model, prompt_enc, embed_dim)
