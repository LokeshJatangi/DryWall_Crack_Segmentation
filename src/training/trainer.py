"""
Training loop with early stopping and TensorBoard logging.
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import PROMPT_TO_ID
from src.training.losses import get_loss
from src.training.metrics import SegmentationMetrics
from src.models.smp_models import create_smp_model
from src.models.segformer import SegFormerFiLM


# Reverse mapping for prediction filenames
ID_TO_PROMPT = {v: k for k, v in PROMPT_TO_ID.items()}


def create_model(config):
    """Create model based on config."""
    if config.model.arch == "SegFormer":
        return SegFormerFiLM(
            num_prompts=config.model.num_prompts,
            embed_dim=config.model.embed_dim,
        )
    else:
        return create_smp_model(
            arch=config.model.arch,
            encoder_name=config.model.encoder,
            num_prompts=config.model.num_prompts,
            embed_dim=config.model.embed_dim,
        )


class Trainer:
    """Handles model training, validation, and prediction saving."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model = create_model(config).to(self.device)

        # Loss
        self.criterion = get_loss(config)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        # Scheduler (onecycle is created in fit() since it needs steps_per_epoch)
        self.scheduler = None
        if config.training.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.training.epochs,
                eta_min=config.training.lr * 0.01,
            )
        elif config.training.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='max', patience=5, factor=0.5,
            )

        # Metrics
        self.metrics = SegmentationMetrics()

        # Experiment directories: experiments/{exp_name}_{timestamp}/logs|ckpts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config.output.exp_name}_{timestamp}"
        run_dir = Path("experiments") / run_name
        self.log_dir = run_dir / "logs"
        self.checkpoint_dir = run_dir / "ckpts"
        self.predictions_dir = run_dir / "predictions"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Checkpointing state
        self.best_dice = 0.0
        self.patience_counter = 0

        self.start_epoch = 1

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: {config.model.arch}")
        print(f"Device: {self.device}")
        print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"Run directory: {run_dir}")

    def train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        """Run one training epoch."""
        self.model.train()
        self.metrics.reset()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
        for batch in pbar:
            images = batch['image'].to(self.device)
            prompt_ids = batch['prompt_id'].to(self.device)
            masks = batch['mask'].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(images, prompt_ids)
            loss = self.criterion(pred.squeeze(1), masks)
            loss.backward()
            self.optimizer.step()

            # OneCycleLR steps per batch
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.metrics.update(pred.squeeze(1), masks)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Log loss every 15 steps
            global_step = (epoch - 1) * len(loader) + num_batches
            if num_batches % 15 == 0:
                self.writer.add_scalar("train/step_loss", loss.item(), global_step)

        metrics = self.metrics.compute()
        metrics['loss'] = total_loss / max(num_batches, 1)
        return metrics

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        """Run validation."""
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(loader, desc="Validating"):
            images = batch['image'].to(self.device)
            prompt_ids = batch['prompt_id'].to(self.device)
            masks = batch['mask'].to(self.device)

            pred = self.model(images, prompt_ids)
            loss = self.criterion(pred.squeeze(1), masks)

            total_loss += loss.item()
            num_batches += 1
            self.metrics.update(pred.squeeze(1), masks)

        metrics = self.metrics.compute()
        metrics['loss'] = total_loss / max(num_batches, 1)
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Full training loop with early stopping and checkpointing."""
        epochs = self.config.training.epochs
        patience = self.config.training.early_stopping_patience
        end_epoch = self.start_epoch + epochs - 1

        # Create OneCycleLR here since it needs steps_per_epoch
        if self.config.training.scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.training.lr,
                steps_per_epoch=len(train_loader),
                epochs=epochs,
            )

        print(f"\nStarting training for {epochs} epochs (epoch {self.start_epoch} → {end_epoch})...")
        print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
        print("-" * 70)

        for epoch in range(self.start_epoch, end_epoch + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            # Update scheduler (OneCycleLR steps per batch in train_epoch)
            if self.scheduler and not isinstance(self.scheduler, OneCycleLR):
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['dice'])
                else:
                    self.scheduler.step()

            # Log to TensorBoard
            for key, val in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", val, epoch)
            for key, val in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", val, epoch)
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], epoch)

            # Print epoch summary
            print(
                f"Epoch {epoch}/{end_epoch} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, "
                f"mIoU: {val_metrics['miou']:.4f}"
            )

            # Checkpoint on improvement
            if val_metrics['dice'] > self.best_dice:
                self.best_dice = val_metrics['dice']
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics)
                print(f"  -> New best model! Dice: {self.best_dice:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\nEarly stopping after {epoch} epochs (no improvement for {patience} epochs)")
                    break

        self.writer.close()
        print(f"\nTraining complete. Best val Dice: {self.best_dice:.4f}")

    def _save_checkpoint(self, epoch: int, metrics: dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)

        best_path = self.checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)

    @torch.no_grad()
    def save_predictions(self, loader: DataLoader, output_dir: str = None) -> None:
        """Save prediction masks as PNG files with values {0, 255}."""
        self.model.eval()
        out_path = Path(output_dir) if output_dir else self.predictions_dir
        out_path.mkdir(parents=True, exist_ok=True)

        sample_idx = 0
        for batch in tqdm(loader, desc="Saving predictions"):
            images = batch['image'].to(self.device)
            prompt_ids = batch['prompt_id'].to(self.device)

            pred = self.model(images, prompt_ids)
            pred_masks = (torch.sigmoid(pred.squeeze(1)) >= 0.5).cpu().numpy().astype(np.uint8) * 255

            for i in range(pred_masks.shape[0]):
                prompt_text = ID_TO_PROMPT[batch['prompt_id'][i].item()]
                prompt_slug = prompt_text.replace(" ", "_")
                filename = f"{sample_idx}__{prompt_slug}.png"
                cv2.imwrite(str(out_path / filename), pred_masks[i])
                sample_idx += 1

        print(f"Saved {sample_idx} predictions to {output_dir}")
