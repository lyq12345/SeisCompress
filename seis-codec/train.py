import argparse
import json
import random
import traceback
from pathlib import Path
import sys

_SEISLM_SRC = Path(__file__).resolve().parents[1] / "seisLM"
if _SEISLM_SRC.exists() and str(_SEISLM_SRC) not in sys.path:
    sys.path.insert(0, str(_SEISLM_SRC))

from typing import Dict, List

import lightning as L
import numpy as np
import seisbench.generate as sbg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import ml_collections
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from seisbench.generate.augmentation import Normalize
from torch.utils.data import DataLoader

# Import SeisDAC that we just created
from model import SeisDAC

# Import Discriminator from descript-audio-codec
from dac.model.discriminator import Discriminator

# Import data loaders from seisLM
from seisLM.data_pipeline import pretrain_dataloaders as dataloaders
from seisLM.utils.data_utils import phase_dict


def waveform_collator(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Stack fixed-length waveform windows into a batch tensor."""
    waveforms = np.stack([sample["X"] for sample in batch])
    return {"waveforms": torch.from_numpy(waveforms)}


def _rebuild_validation_dataloaders(
    dev_loaders: Dict[str, DataLoader],
    *,
    num_workers: int,
    prefetch_factor: int = 2,
) -> Dict[str, DataLoader]:
    """Rebuild validation loaders with a validation-specific worker count.

    The seisLM helper uses the same ``num_workers`` for train and validation.
    For deterministic validation with random window augmentations, validation
    samples should be generated in the main process after reseeding at the start
    of every validation epoch, while keeping the training loader parallel.
    """
    rebuilt = {}
    for name, loader in dev_loaders.items():
        kwargs = {
            "batch_size": loader.batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": loader.pin_memory,
            "collate_fn": loader.collate_fn,
            "drop_last": False,
        }
        if num_workers > 0:
            kwargs["worker_init_fn"] = loader.worker_init_fn
            kwargs["prefetch_factor"] = prefetch_factor
        rebuilt[name] = DataLoader(loader.dataset, **kwargs)
    return rebuilt


def _json_safe_score(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    try:
        return float(value)
    except TypeError:
        return str(value)


class SeismicSTFTLoss(nn.Module):
    def __init__(self, window_lengths=[256, 128, 64, 32]):
        super().__init__()
        self.window_lengths = window_lengths

    def forward(self, x, y):
        B, C, T = x.shape
        x = x.reshape(B * C, T)
        y = y.reshape(B * C, T)
        
        loss = 0.0
        for w in self.window_lengths:
            hop_length = w // 4
            window = torch.hann_window(w).to(x.device)
            x_stft = torch.stft(x, n_fft=w, hop_length=hop_length, window=window, return_complex=True)
            y_stft = torch.stft(y, n_fft=w, hop_length=hop_length, window=window, return_complex=True)
            
            x_mag = torch.abs(x_stft) + 1e-5
            y_mag = torch.abs(y_stft) + 1e-5
            
            loss += F.l1_loss(x_mag, y_mag) + F.l1_loss(torch.log10(x_mag), torch.log10(y_mag))
            
        return loss

class SeisDACLightning(L.LightningModule):
    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Instantiate Generator
        self.generator = SeisDAC(
            in_channels=config.model.in_channels,
            sample_rate=config.model.sample_rate,
            encoder_dim=config.model.encoder_dim,
            decoder_dim=config.model.decoder_dim,
            encoder_rates=config.model.encoder_rates,
            decoder_rates=config.model.decoder_rates,
            use_stable_quantizer=config.model.get("use_stable_quantizer", True),
            use_seislm_encoder=config.model.get("use_seislm_encoder", False),
            seislm_encoder_checkpoint=config.model.get("seislm_encoder_checkpoint", ""),
            freeze_seislm_extractor=config.model.get("freeze_seislm_extractor", False),
        )

        self.use_gan = config.training.get("use_gan", True)
        self.discriminator = Discriminator() if self.use_gan else None

        # Manual optimization is only required for GAN training.
        self.automatic_optimization = not self.use_gan

        # Task-aware Loss Setup (SeisLM)
        self.use_task_aware_loss = config.training.get('use_task_aware_loss', False)
        self.task_aware_weight = config.training.get('task_aware_weight', 1.0)
        self.seis_lm_model = None

        # Spectral Loss Setup
        self.use_spectral_loss = config.training.get('use_spectral_loss', False)
        self.spectral_weight = config.training.get('spectral_weight', 1.0)
        if self.use_spectral_loss:
            self.stft_loss = SeismicSTFTLoss(window_lengths=config.training.get('stft_window_lengths', [256, 128, 64, 32]))

        if self.use_task_aware_loss:
            checkpoint_path = config.training.get('seis_lm_checkpoint', None)
            if checkpoint_path:
                from seisLM.model.foundation.pretrained_models import LitMultiDimWav2Vec2

                print(f"Loading SeisLM from {checkpoint_path} for Task-aware Loss...")
                self.seis_lm_model = LitMultiDimWav2Vec2.load_from_checkpoint(checkpoint_path)
            else:
                raise ValueError("seis_lm_checkpoint must be provided if use_task_aware_loss is True.")
            
            # Freeze the SeisLM model
            self.seis_lm_model.eval()
            for param in self.seis_lm_model.parameters():
                param.requires_grad = False

        self.val_dataloader_names: List[str] = []
        self.gradient_clip_g = config.training.get("gradient_clip_g", 1000.0)
        self.gradient_clip_d = config.training.get("gradient_clip_d", 10.0)
        self.feature_matching_weight = config.training.get("feature_matching_weight", 2.0)
        self.latent_reg_weight = float(config.training.get("latent_reg_weight", 0.0))
        self.latent_reg_threshold = float(config.training.get("latent_reg_threshold", 1.0))
        self.latent_reg_target = config.training.get("latent_reg_target", "latents")

    def _log_vq_latent_stats(self, out: Dict) -> None:
        with torch.no_grad():
            latents = out["latents"]
            encoder_latent = out["encoder_latent"]

            latent_norm = latents.norm(dim=1)
            encoder_norm = encoder_latent.norm(dim=1)

            self.log("train/latent_norm_mean", latent_norm.mean())
            self.log("train/latent_norm_max", latent_norm.max())
            self.log("train/latent_absmax", latents.abs().max())
            self.log("train/encoder_latent_norm_mean", encoder_norm.mean())
            self.log("train/encoder_latent_norm_max", encoder_norm.max())
            self.log("train/encoder_latent_absmax", encoder_latent.abs().max())

    def _compute_latent_regularization_loss(self, out: Dict) -> torch.Tensor:
        if self.latent_reg_weight <= 0:
            return torch.tensor(0.0, device=self.device)

        if self.latent_reg_target == "both":
            target_names = ("latents", "encoder_latent")
        else:
            target_names = (self.latent_reg_target,)

        loss = torch.tensor(0.0, device=self.device)
        for name in target_names:
            x = out[name]
            rms = x.pow(2).mean(dim=1).sqrt()
            excess = F.relu(rms - self.latent_reg_threshold)
            loss = loss + excess.pow(2).mean()
        return loss

    def forward(self, x):
        return self.generator(x)

    def _prepare_waveforms(self, batch) -> torch.Tensor:
        real_waveforms = batch["waveforms"] if isinstance(batch, dict) else batch[0]
        if real_waveforms.ndim == 2:
            real_waveforms = real_waveforms.unsqueeze(1)
        return real_waveforms

    def _forward(self, real_waveforms: torch.Tensor):
        out = self.generator(real_waveforms, sample_rate=self.config.model.sample_rate)
        return out["audio"], out

    def _compute_reconstruction_losses(self, fake_waveforms, real_waveforms):
        loss_l1 = F.l1_loss(fake_waveforms, real_waveforms)

        loss_task = torch.tensor(0.0, device=self.device)
        if self.use_task_aware_loss:
            loss_task = self.compute_task_aware_loss(fake_waveforms, real_waveforms)

        loss_spectral = torch.tensor(0.0, device=self.device)
        if self.use_spectral_loss:
            loss_spectral = self.stft_loss(fake_waveforms, real_waveforms)

        return loss_l1, loss_task, loss_spectral

    def _val_reconstruction_loss(self, loss_l1, loss_task, loss_spectral):
        return (
            100.0 * loss_l1
            + self.task_aware_weight * loss_task
            + self.spectral_weight * loss_spectral
        )

    def _normalize_augmentation(self) -> Normalize:
        data_config = self.config.data
        return Normalize(
            demean_axis=data_config.get("demean_axis", None),
            amp_norm_axis=data_config.get("amp_norm_axis", None),
            amp_norm_type=data_config.get("amp_norm_type", "peak"),
        )

    def _seed_validation_rng(self) -> None:
        if not self.config.data.get("deterministic_val", True):
            return

        self._validation_rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        seed = int(self.config.data.get("val_seed", self.config.seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _restore_validation_rng(self) -> None:
        state = getattr(self, "_validation_rng_state", None)
        if state is None:
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
        self._validation_rng_state = None

    def on_validation_epoch_start(self) -> None:
        self._seed_validation_rng()

    def on_validation_epoch_end(self) -> None:
        self._restore_validation_rng()

    def get_train_augmentations(self) -> List:
        return [
            sbg.WindowAroundSample(
                list(phase_dict.keys()),
                samples_before=3000,
                windowlen=6000,
                selection="random",
                strategy="variable",
            ),
            sbg.RandomWindow(
                low=None,
                high=None,
                windowlen=self.config.data.window_length,
                strategy="pad",
            ),
            sbg.ChangeDtype(np.float32),
            self._normalize_augmentation(),
        ]

    def get_val_augmentations(self) -> List:
        return [
            sbg.WindowAroundSample(
                list(phase_dict.keys()),
                samples_before=3000,
                windowlen=6000,
                selection="random",
                strategy="variable",
            ),
            sbg.RandomWindow(
                low=None,
                high=None,
                windowlen=self.config.data.window_length,
                strategy="pad",
            ),
            sbg.ChangeDtype(np.float32),
            self._normalize_augmentation(),
        ]

    def configure_optimizers(self):
        opt_g = AdamW(self.generator.parameters(), lr=self.config.training.learning_rate, betas=(0.8, 0.99))
        if not self.use_gan:
            return opt_g

        opt_d = AdamW(self.discriminator.parameters(), lr=self.config.training.learning_rate, betas=(0.8, 0.99))
        return [opt_g, opt_d], []

    def compute_adv_loss(self, fake, real):
        # fake and real shape: (B, C, T)
        # Reshape to (B*C, 1, T) to evaluate all channels with the 1-channel Discriminator
        B, C, T = fake.shape
        fake_d = fake.reshape(B * C, 1, T)
        real_d = real.reshape(B * C, 1, T)

        d_fake = self.discriminator(fake_d)
        d_real = self.discriminator(real_d)
        return d_fake, d_real

    def discriminator_loss(self, d_fake, d_real):
        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)
        return loss_d

    def generator_loss(self, d_fake, d_real):
        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0
        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
        return loss_g, loss_feature

    def compute_task_aware_loss(self, fake_waveforms, real_waveforms):
        if not self.seis_lm_model:
            return torch.tensor(0.0).to(self.device)
        
        # We need to extract features from the frozen SeisLM model.
        # Assuming we can pass waveforms directly to the underlying model's feature extractor.
        # SeisLM inner model is MultiDimWav2Vec2ForPreTraining which has wav2vec2.feature_extractor
        try:
            with torch.no_grad():
                real_features = self.seis_lm_model.model.wav2vec2.feature_extractor(real_waveforms)
            fake_features = self.seis_lm_model.model.wav2vec2.feature_extractor(fake_waveforms)
            
            # Use L1 distance between the extracted feature maps
            task_loss = F.l1_loss(fake_features, real_features)
            return task_loss
        except Exception as e:
            # Fallback if the feature extraction logic varies
            print(f"Failed to compute task-aware loss: {e}")
            return torch.tensor(0.0).to(self.device)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data_name = self.val_dataloader_names[dataloader_idx]
        real_waveforms = self._prepare_waveforms(batch)
        fake_waveforms, _ = self._forward(real_waveforms)
        loss_l1, loss_task, loss_spectral = self._compute_reconstruction_losses(
            fake_waveforms, real_waveforms
        )
        val_loss = self._val_reconstruction_loss(loss_l1, loss_task, loss_spectral)

        logs = {
            f"val/l1/{data_name}": loss_l1,
            f"val/loss/{data_name}": val_loss,
        }
        if self.use_task_aware_loss:
            logs[f"val/loss_task/{data_name}"] = loss_task
        if self.use_spectral_loss:
            logs[f"val/loss_spectral/{data_name}"] = loss_spectral

        self.log_dict(
            logs,
            on_step=False,
            on_epoch=True,
            prog_bar=dataloader_idx == 0,
            add_dataloader_idx=False,
        )

    def training_step(self, batch, batch_idx):
        real_waveforms = self._prepare_waveforms(batch)
        fake_waveforms, out = self._forward(real_waveforms)
        commitment_loss = out["vq/commitment_loss"]
        codebook_loss = out["vq/codebook_loss"]
        loss_l1, loss_task, loss_spectral = self._compute_reconstruction_losses(
            fake_waveforms, real_waveforms
        )
        loss_latent_reg = self._compute_latent_regularization_loss(out)

        if not self.use_gan:
            loss_recon = 100.0 * loss_l1
            loss_vq = 0.25 * commitment_loss + 1.0 * codebook_loss
            loss = (
                loss_recon
                + loss_vq
                + self.task_aware_weight * loss_task
                + self.spectral_weight * loss_spectral
                + self.latent_reg_weight * loss_latent_reg
            )
            self.log("train/loss", loss, prog_bar=True)
            self.log("train/loss_recon", loss_recon, prog_bar=True)
            self.log("train/loss_vq", loss_vq)
            self.log("train/l1", loss_l1)
            self.log("train/commitment", commitment_loss)
            self.log("train/codebook", codebook_loss)
            if self.use_task_aware_loss:
                self.log("train/loss_task", loss_task)
            if self.use_spectral_loss:
                self.log("train/loss_spectral", loss_spectral)
            if self.latent_reg_weight > 0:
                self.log("train/loss_latent_reg", loss_latent_reg)
                self.log("train/loss_latent_reg_weighted", self.latent_reg_weight * loss_latent_reg)
            self._log_vq_latent_stats(out)
            return loss

        opt_g, opt_d = self.optimizers()

        # Train Discriminator
        self.toggle_optimizer(opt_d)
        d_fake, d_real = self.compute_adv_loss(fake_waveforms.detach(), real_waveforms)
        loss_d = self.discriminator_loss(d_fake, d_real)
        self.manual_backward(loss_d)
        self.clip_gradients(opt_d, gradient_clip_val=self.gradient_clip_d)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # Train Generator
        self.toggle_optimizer(opt_g)
        d_fake, d_real = self.compute_adv_loss(fake_waveforms, real_waveforms)
        loss_g_adv, loss_feature = self.generator_loss(d_fake, d_real)

        loss_g = (loss_g_adv +
                  self.feature_matching_weight * loss_feature +
                  100.0 * loss_l1 +
                  0.25 * commitment_loss +
                  1.0 * codebook_loss +
                  self.task_aware_weight * loss_task +
                  self.spectral_weight * loss_spectral +
                  self.latent_reg_weight * loss_latent_reg)

        self.manual_backward(loss_g)
        self.clip_gradients(opt_g, gradient_clip_val=self.gradient_clip_g)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        self.log("train/loss_g", loss_g, prog_bar=True)
        self.log("train/loss_d", loss_d, prog_bar=True)
        self.log("train/loss_g_adv", loss_g_adv)
        self.log("train/loss_feature", loss_feature)
        self.log("train/l1", loss_l1)
        self.log("train/commitment", commitment_loss)
        self.log("train/codebook", codebook_loss)
        if self.use_task_aware_loss:
            self.log("train/loss_task", loss_task)
        if self.use_spectral_loss:
            self.log("train/loss_spectral", loss_spectral)
        if self.latent_reg_weight > 0:
            self.log("train/loss_latent_reg", loss_latent_reg)
            self.log("train/loss_latent_reg_weighted", self.latent_reg_weight * loss_latent_reg)
        self._log_vq_latent_stats(out)

def train(config):
    L.seed_everything(config.seed)
    model = SeisDACLightning(config)

    train_loader, dev_loaders = dataloaders.prepare_pretrain_dataloaders(
        model=model,
        training_fraction=config.data.training_fraction,
        data_names=config.data.data_name,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        prefetch_factor=2,
        cache=config.data.cache_dataset,
        collator=waveform_collator,
        include_shock_val=config.data.get("include_shock_val", False),
    )

    val_num_workers = int(config.data.get("val_num_workers", config.data.num_workers))
    if val_num_workers != config.data.num_workers:
        dev_loaders = _rebuild_validation_dataloaders(
            dev_loaders,
            num_workers=val_num_workers,
            prefetch_factor=2,
        )
        print(f"Validation DataLoader num_workers set to {val_num_workers}")

    model.val_dataloader_names = list(dev_loaders.keys())
    if not dev_loaders:
        raise ValueError("No validation dataloaders configured.")
    print(f"Validation sets: {', '.join(dev_loaders.keys())}")

    log_dir = Path(config.training.get("log_dir", "lightning_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = TensorBoardLogger(
        save_dir=str(log_dir),
        name=config.training.get("log_name", "seisdac"),
        version=config.training.get("log_version", None),
    )
    print(f"Logging to: {logger.log_dir}")

    monitor_metric = config.training.get("monitor_metric", "loss")
    if monitor_metric.startswith("val/"):
        primary_val_metric = monitor_metric
    else:
        primary_val_metric = f"val/{monitor_metric}/{model.val_dataloader_names[0]}"
    print(f"Checkpoint monitor: {primary_val_metric} (mode=min)")

    checkpoint_callback = ModelCheckpoint(
        monitor=primary_val_metric,
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch}-{step}",
        auto_insert_metric_name=False,
    )

    callbacks = [checkpoint_callback]
    early_stopping_patience = int(config.training.get("early_stopping_patience", 0))
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=primary_val_metric,
                mode="min",
                patience=early_stopping_patience,
                min_delta=float(config.training.get("early_stopping_min_delta", 0.0)),
                strict=True,
            )
        )
        print(
            "Early stopping enabled: "
            f"patience={early_stopping_patience}, "
            f"min_delta={config.training.get('early_stopping_min_delta', 0.0)}"
        )

    num_devices = config.training.get("devices", 1)
    use_gpu = torch.cuda.is_available()
    if use_gpu and num_devices == -1:
        num_devices = torch.cuda.device_count()
    strategy = "auto"
    if use_gpu and num_devices > 1:
        # GAN alternates D/G optimizers; some params are unused each step.
        strategy = "ddp_find_unused_parameters_true"
        print(f"Training on {num_devices} GPUs with strategy={strategy}")

    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        devices=num_devices if use_gpu else 1,
        accelerator="gpu" if use_gpu else "cpu",
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=None if config.training.use_gan else config.training.get("gradient_clip_g", 1000.0),
    )

    trainer.fit(model, train_loader, list(dev_loaders.values()))

    checkpoint_dir = Path(logger.log_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint_path = None
    if config.training.get("save_final_checkpoint", True):
        final_checkpoint_path = checkpoint_dir / "final.ckpt"
        trainer.save_checkpoint(str(final_checkpoint_path))

    if trainer.is_global_zero:
        summary = {
            "monitor": primary_val_metric,
            "mode": "min",
            "best_model_path": checkpoint_callback.best_model_path or None,
            "best_model_score": _json_safe_score(checkpoint_callback.best_model_score),
            "last_model_path": checkpoint_callback.last_model_path or None,
            "final_model_path": str(final_checkpoint_path) if final_checkpoint_path else None,
        }
        summary_path = checkpoint_dir / "checkpoint_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"Checkpoint summary saved to: {summary_path}")
        print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_run", action="store_true")
    parser.add_argument(
        "--no_gan",
        action="store_true",
        help="Disable GAN/discriminator and train with reconstruction + VQ loss only.",
    )
    # Task-aware loss toggle via command line
    parser.add_argument("--use_task_aware_loss", action="store_true", help="Enable SeisLM task-aware loss")
    parser.add_argument("--seis_lm_checkpoint", type=str, default="", help="Path to SeisLM pretrained checkpoint")
    parser.add_argument("--use_spectral_loss", action="store_true", help="Enable Multi-scale STFT spectral loss")
    parser.add_argument(
        "--amp_norm_type",
        choices=["peak", "std", "none"],
        default="peak",
        help="Waveform amplitude normalization along time axis. Use 'none' for raw amplitudes.",
    )
    parser.add_argument(
        "--no_demean",
        action="store_true",
        help="Disable per-channel time-axis demeaning before amplitude normalization.",
    )
    parser.add_argument(
        "--log_name",
        type=str,
        default="seisdac",
        help="TensorBoard experiment name under log_dir/.",
    )
    parser.add_argument(
        "--log_version",
        type=str,
        default="",
        help="TensorBoard run version (e.g. v1). Leave empty for auto version_0, version_1, ...",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="lightning_logs",
        help="Root directory for TensorBoard logs and checkpoints.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Per-GPU batch size (default: 32, or 4 with --test_run). Use 4-8 on 16GB GPUs with GAN+spectral loss.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs (default: 1). Use -1 for all available GPUs.",
    )
    parser.add_argument(
        "--include_shock_val",
        action="store_true",
        help="Also validate on foreshock/aftershock shock data (requires data/foreshock_aftershock_NRCA/). ETHZ dev split is always used for validation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Base learning rate for generator and discriminator optimizers.",
    )
    parser.add_argument(
        "--gradient_clip_g",
        type=float,
        default=1000.0,
        help="Generator gradient clipping value. Lower values can reduce late-training latent drift.",
    )
    parser.add_argument(
        "--gradient_clip_d",
        type=float,
        default=10.0,
        help="Discriminator gradient clipping value.",
    )
    parser.add_argument(
        "--feature_matching_weight",
        type=float,
        default=2.0,
        help="Weight for discriminator feature matching in the generator loss.",
    )
    parser.add_argument(
        "--latent_reg_weight",
        type=float,
        default=0.0,
        help="Weight for latent magnitude regularization. Use 0 to disable.",
    )
    parser.add_argument(
        "--latent_reg_threshold",
        type=float,
        default=1.0,
        help="Per-time-step channel RMS threshold before latent regularization is applied.",
    )
    parser.add_argument(
        "--latent_reg_target",
        choices=["latents", "encoder_latent", "both"],
        default="latents",
        help="Tensor to regularize for latent magnitude control.",
    )
    parser.add_argument(
        "--no_stable_vq",
        action="store_true",
        help="Use the original DAC quantizer instead of quantize_stable.py.",
    )
    parser.add_argument(
        "--use_seislm_encoder",
        action="store_true",
        help="Replace the DAC encoder with the pretrained SeisLM feature extractor + adapter (Plan A).",
    )
    parser.add_argument(
        "--seislm_encoder_checkpoint",
        type=str,
        default="/scratch/yuqiao-models/seisLM/pretrained_seislm_base/checkpoints/epoch=39-step=1203000.ckpt",
        help="Pretrained SeisLM checkpoint used to initialize the encoder.",
    )
    parser.add_argument(
        "--freeze_seislm_extractor",
        action="store_true",
        help="Freeze the pretrained SeisLM conv layers (only train the adapter).",
    )
    parser.add_argument(
        "--monitor_metric",
        choices=["loss", "l1"],
        default="loss",
        help="Validation metric suffix to monitor: val/loss/<dataset> or val/l1/<dataset>.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help="Stop after this many validation epochs without monitor improvement. Use 0 to disable.",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help="Minimum monitor improvement required by EarlyStopping.",
    )
    parser.add_argument(
        "--disable_early_stopping",
        action="store_true",
        help="Disable EarlyStopping regardless of --early_stopping_patience.",
    )
    parser.add_argument(
        "--no_deterministic_val",
        action="store_true",
        help="Do not reseed RNG before each validation epoch.",
    )
    parser.add_argument(
        "--val_seed",
        type=int,
        default=42,
        help="RNG seed used for deterministic validation augmentations.",
    )
    parser.add_argument(
        "--val_num_workers",
        type=int,
        default=0,
        help="Validation DataLoader workers. Keep 0 for deterministic random-window validation.",
    )
    parser.add_argument(
        "--no_save_final_checkpoint",
        action="store_true",
        help="Do not save checkpoints/final.ckpt at the end of training.",
    )
    args = parser.parse_args()

    config = ml_collections.ConfigDict({
        "seed": 42,
        "model": {
            "in_channels": 3,
            "sample_rate": 100,
            "encoder_dim": 64,
            "decoder_dim": 1536,
            "encoder_rates": [2, 2, 2],
            "decoder_rates": [2, 2, 2],
            "use_stable_quantizer": not args.no_stable_vq,
            "use_seislm_encoder": args.use_seislm_encoder,
            "seislm_encoder_checkpoint": args.seislm_encoder_checkpoint if args.use_seislm_encoder else "",
            "freeze_seislm_extractor": args.freeze_seislm_extractor,
        },
        "training": {
            "learning_rate": args.learning_rate,
            "max_epochs": 1 if args.test_run else 100,
            "use_gan": not args.no_gan,
            "use_task_aware_loss": args.use_task_aware_loss,
            "seis_lm_checkpoint": args.seis_lm_checkpoint,
            "task_aware_weight": 10.0,
            "use_spectral_loss": args.use_spectral_loss,
            "gradient_clip_g": args.gradient_clip_g,
            "gradient_clip_d": args.gradient_clip_d,
            "feature_matching_weight": args.feature_matching_weight,
            "latent_reg_weight": args.latent_reg_weight,
            "latent_reg_threshold": args.latent_reg_threshold,
            "latent_reg_target": args.latent_reg_target,
            "devices": args.devices,
            "log_dir": args.log_dir,
            "log_name": args.log_name,
            "log_version": args.log_version or None,
            "monitor_metric": args.monitor_metric,
            "early_stopping_patience": 0 if args.disable_early_stopping else args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "save_final_checkpoint": not args.no_save_final_checkpoint,
        },
        "data": {
            "data_name": ["ETHZ"],
            "batch_size": args.batch_size or (4 if args.test_run else 32),
            "training_fraction": 0.1 if args.test_run else 1.0,
            "num_workers": 2,
            "val_num_workers": args.val_num_workers,
            "deterministic_val": not args.no_deterministic_val,
            "val_seed": args.val_seed,
            "cache_dataset": None,
            "window_length": 3001,
            "include_shock_val": args.include_shock_val,
            "demean_axis": None if args.no_demean or args.amp_norm_type == "none" else -1,
            "amp_norm_axis": None if args.amp_norm_type == "none" else -1,
            "amp_norm_type": "peak" if args.amp_norm_type == "none" else args.amp_norm_type,
        }
    })

    train(config)
