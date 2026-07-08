import argparse
import json
import traceback
from typing import Dict, List

import lightning as L
import numpy as np
import seisbench.generate as sbg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import ml_collections
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from seisbench.generate.augmentation import Normalize

# Import SeisDAC that we just created
from model import SeisDAC

# Import Discriminator from descript-audio-codec
from dac.model.discriminator import Discriminator

# Import data loaders from seisLM
from seisLM.data_pipeline import pretrain_dataloaders as dataloaders
from seisLM.model.foundation.pretrained_models import LitMultiDimWav2Vec2
from seisLM.utils.data_utils import phase_dict


def waveform_collator(batch: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
    """Stack fixed-length waveform windows into a batch tensor."""
    waveforms = np.stack([sample["X"] for sample in batch])
    return {"waveforms": torch.from_numpy(waveforms)}

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
            decoder_rates=config.model.decoder_rates
        )

        self.use_gan = config.training.get("use_gan", True)
        self.discriminator = Discriminator() if self.use_gan else None

        # Manual optimization is only required for GAN training.
        self.automatic_optimization = not self.use_gan

        # Task-aware Loss Setup (SeisLM)
        self.use_task_aware_loss = config.training.get('use_task_aware_loss', False)
        self.task_aware_weight = config.training.get('task_aware_weight', 1.0)
        self.seis_lm_model = None

        if self.use_task_aware_loss:
            checkpoint_path = config.training.get('seis_lm_checkpoint', None)
            if checkpoint_path:
                print(f"Loading SeisLM from {checkpoint_path} for Task-aware Loss...")
                self.seis_lm_model = LitMultiDimWav2Vec2.load_from_checkpoint(checkpoint_path)
            else:
                raise ValueError("seis_lm_checkpoint must be provided if use_task_aware_loss is True.")
            
            # Freeze the SeisLM model
            self.seis_lm_model.eval()
            for param in self.seis_lm_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.generator(x)

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
            Normalize(),
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
            Normalize(),
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

    def training_step(self, batch, batch_idx):
        real_waveforms = batch['waveforms'] if isinstance(batch, dict) else batch[0]

        if real_waveforms.ndim == 2:
            real_waveforms = real_waveforms.unsqueeze(1)

        out = self.generator(real_waveforms, sample_rate=self.config.model.sample_rate)
        fake_waveforms = out["audio"]
        commitment_loss = out["vq/commitment_loss"]
        codebook_loss = out["vq/codebook_loss"]
        loss_l1 = F.l1_loss(fake_waveforms, real_waveforms)

        loss_task = torch.tensor(0.0, device=self.device)
        if self.use_task_aware_loss:
            loss_task = self.compute_task_aware_loss(fake_waveforms, real_waveforms)

        if not self.use_gan:
            loss = (
                100.0 * loss_l1
                + 0.25 * commitment_loss
                + 1.0 * codebook_loss
                + self.task_aware_weight * loss_task
            )
            self.log("train/loss", loss, prog_bar=True)
            self.log("train/l1", loss_l1, prog_bar=True)
            self.log("train/commitment", commitment_loss)
            self.log("train/codebook", codebook_loss)
            if self.use_task_aware_loss:
                self.log("train/loss_task", loss_task)
            return loss

        opt_g, opt_d = self.optimizers()

        # Train Discriminator
        self.toggle_optimizer(opt_d)
        d_fake, d_real = self.compute_adv_loss(fake_waveforms.detach(), real_waveforms)
        loss_d = self.discriminator_loss(d_fake, d_real)
        self.manual_backward(loss_d)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # Train Generator
        self.toggle_optimizer(opt_g)
        d_fake, d_real = self.compute_adv_loss(fake_waveforms, real_waveforms)
        loss_g_adv, loss_feature = self.generator_loss(d_fake, d_real)

        loss_g = (loss_g_adv +
                  2.0 * loss_feature +
                  100.0 * loss_l1 +
                  0.25 * commitment_loss +
                  1.0 * codebook_loss +
                  self.task_aware_weight * loss_task)

        self.manual_backward(loss_g)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        self.log("train/loss_g", loss_g, prog_bar=True)
        self.log("train/loss_d", loss_d, prog_bar=True)
        self.log("train/l1", loss_l1)
        if self.use_task_aware_loss:
            self.log("train/loss_task", loss_task)

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
    )
    
    trainer = L.Trainer(
        max_epochs=config.training.max_epochs,
        devices=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    )
    
    trainer.fit(model, train_loader, list(dev_loaders.values()))

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
    args = parser.parse_args()

    config = ml_collections.ConfigDict({
        "seed": 42,
        "model": {
            "in_channels": 3,
            "sample_rate": 100,
            "encoder_dim": 64,
            "decoder_dim": 1536,
            "encoder_rates": [2, 2, 2],
            "decoder_rates": [2, 2, 2]
        },
        "training": {
            "learning_rate": 1e-4,
            "max_epochs": 1 if args.test_run else 100,
            "use_gan": not args.no_gan,
            "use_task_aware_loss": args.use_task_aware_loss,
            "seis_lm_checkpoint": args.seis_lm_checkpoint,
            "task_aware_weight": 10.0 # Can be tuned
        },
        "data": {
            "data_name": ["ETHZ"],
            "batch_size": 4 if args.test_run else 32,
            "training_fraction": 0.1 if args.test_run else 1.0,
            "num_workers": 2,
            "cache_dataset": None,
            "window_length": 3001,
        }
    })

    train(config)
