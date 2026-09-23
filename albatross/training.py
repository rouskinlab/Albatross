#!/usr/bin/env python3
"""
Continued pretraining of RiNALMo on IRES sequences (Albatross).

BERT-style masked language modeling at 15% mask rate with the objective
described in the Methods: corruption and loss positions are sampled
independently. Optimizer and schedule match the paper defaults.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor

import argparse
from pathlib import Path
from typing import List
import random
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rinalmo.config import model_config
from rinalmo.model.model import RiNALMo
from rinalmo.data.alphabet import Alphabet
from rinalmo.data.constants import MASK_TKN, PAD_TKN, CLS_TKN, EOS_TKN


class IRESDataset(Dataset):
    """Dataset for IRES RNA sequences with BERT-style masking for MLM pretraining."""

    def __init__(
        self,
        sequences: List[str],
        alphabet: Alphabet,
        mask_ratio: float = 0.15,
        mask_token_prob: float = 0.8,
        random_token_prob: float = 0.1,
        max_length: int = 2048,
    ):
        self.sequences = sequences
        self.alphabet = alphabet
        self.mask_ratio = mask_ratio
        self.mask_token_prob = mask_token_prob
        self.random_token_prob = random_token_prob
        self.max_length = max_length

        self.mask_idx = alphabet.get_idx(MASK_TKN)
        self.pad_idx = alphabet.get_idx(PAD_TKN)
        self.cls_idx = alphabet.get_idx(CLS_TKN)
        self.eos_idx = alphabet.get_idx(EOS_TKN)

        self.rna_token_indices = [alphabet.get_idx(token) for token in ["A", "C", "G", "T"]]

        print(f"Loaded {len(self.sequences)} IRES sequences")
        print(f"Mask ratio: {self.mask_ratio}")
        print(f"Max length: {self.max_length}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        sequence = sequence.upper().replace("U", "T")

        if len(sequence) > self.max_length - 2:
            sequence = sequence[: self.max_length - 2]

        tokens = [self.cls_idx] + [self.alphabet.get_idx(tkn) for tkn in sequence] + [self.eos_idx]
        tokens = torch.tensor(tokens, dtype=torch.long)

        masked_tokens, labels = self._create_masked_lm_predictions(tokens)

        return {
            "input_ids": masked_tokens,
            "labels": labels,
            "attention_mask": torch.ones_like(masked_tokens),
        }

    def _create_masked_lm_predictions(self, tokens):
        """Create masked tokens and labels for MLM following BERT approach."""
        tokens = tokens.clone()
        labels = tokens.clone()

        special_tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
        special_tokens_mask[0] = True  # CLS
        special_tokens_mask[-1] = True  # EOS

        num_to_mask = max(1, int(len(tokens) * self.mask_ratio))

        candidate_positions = torch.where(~special_tokens_mask)[0]

        if len(candidate_positions) > 0:
            mask_positions = candidate_positions[
                torch.randperm(len(candidate_positions))[:num_to_mask]
            ]

            for pos in mask_positions:
                prob = random.random()
                if prob < self.mask_token_prob:
                    tokens[pos] = self.mask_idx
                elif prob < self.mask_token_prob + self.random_token_prob:
                    tokens[pos] = random.choice(self.rna_token_indices)

        mask = torch.zeros_like(labels, dtype=torch.bool)
        if len(candidate_positions) > 0 and num_to_mask > 0:
            mask_positions = candidate_positions[
                torch.randperm(len(candidate_positions))[:num_to_mask]
            ]
            mask[mask_positions] = True

        labels[~mask] = -100

        return tokens, labels


def collate_fn(batch):
    """Collate function to handle variable length sequences."""
    max_len = max(len(item["input_ids"]) for item in batch)

    batch_size = len(batch)
    pad_idx = 1  # PAD token index

    input_ids = torch.full((batch_size, max_len), fill_value=pad_idx, dtype=torch.long)
    labels = torch.full((batch_size, max_len), fill_value=-100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])
        input_ids[i, :seq_len] = item["input_ids"]
        labels[i, :seq_len] = item["labels"]
        attention_mask[i, :seq_len] = item["attention_mask"]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


class IRESPretrainingWrapper(pl.LightningModule):
    """PyTorch Lightning wrapper for IRES pretraining."""

    def __init__(
        self,
        lm_config: str = "giga",
        lr: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = RiNALMo(model_config(lm_config))

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        print(
            f"Initialized IRES pretraining model with "
            f"{sum(p.numel() for p in self.model.parameters()):,} parameters"
        )

    def load_pretrained_weights(self, pretrained_weights_path):
        """Load pretrained RiNALMo weights."""
        print(f"Loading pretrained weights from {pretrained_weights_path}")
        state_dict = torch.load(pretrained_weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict)
        print("Pretrained weights loaded successfully")

    def forward(self, input_ids, attention_mask=None):
        """Forward pass through RiNALMo."""
        outputs = self.model(input_ids)
        return outputs["logits"]

    def training_step(self, batch, batch_idx):
        """Training step with MLM loss."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        logits = self(input_ids, attention_mask)

        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        loss = self.loss_fn(logits_flat, labels_flat)

        with torch.no_grad():
            mask = labels_flat != -100
            if mask.sum() > 0:
                pred_tokens = logits_flat.argmax(dim=-1)
                correct = (pred_tokens == labels_flat) & mask
                accuracy = correct.sum().float() / mask.sum().float()
            else:
                accuracy = torch.tensor(0.0)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/perplexity", torch.exp(loss), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        logits = self(input_ids, attention_mask)

        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        loss = self.loss_fn(logits_flat, labels_flat)

        with torch.no_grad():
            mask = labels_flat != -100
            if mask.sum() > 0:
                pred_tokens = logits_flat.argmax(dim=-1)
                correct = (pred_tokens == labels_flat) & mask
                accuracy = correct.sum().float() / mask.sum().float()
            else:
                accuracy = torch.tensor(0.0)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/perplexity", torch.exp(loss), on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warmup_scheduler,
                "interval": "step",
            },
        }


def load_sequences_from_file(file_path: str) -> List[str]:
    """Load RNA sequences from FASTA or text file."""
    sequences = []
    file_path = Path(file_path)

    print(f"Loading sequences from {file_path}")

    if file_path.suffix.lower() in [".fasta", ".fa", ".fas"]:
        current_seq = ""
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        sequences.append(current_seq)
                        current_seq = ""
                else:
                    current_seq += line
            if current_seq:
                sequences.append(current_seq)
    else:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    sequences.append(line)

    print(f"Loaded {len(sequences)} sequences")

    valid_sequences = []
    for i, seq in enumerate(sequences):
        seq = seq.upper().replace("U", "T")
        if all(c in "ACGTN" for c in seq):
            valid_sequences.append(seq)
        else:
            print(f"Warning: Skipping invalid sequence {i + 1}: contains non-RNA characters")

    print(f"Validated {len(valid_sequences)} sequences")
    return valid_sequences


def main():
    parser = argparse.ArgumentParser(
        description="Continue pretraining RiNALMo on IRES sequences (Albatross)"
    )

    parser.add_argument(
        "data_path", type=str, help="Path to IRES sequences file (FASTA or text)"
    )
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")

    parser.add_argument("--lm_config", type=str, default="giga", help="RiNALMo model configuration")
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default="./weights/rinalmo_giga_pretrained.pt",
        help="Path to pretrained RiNALMo weights",
    )

    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum epochs")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (default: None, use max_epochs only)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10000,
        help="Warmup steps (paper default: 10000)",
    )
    parser.add_argument(
        "--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value"
    )

    parser.add_argument("--mask_ratio", type=float, default=0.15, help="Ratio of tokens to mask")
    parser.add_argument(
        "--mask_token_prob", type=float, default=0.8, help="Probability of using [MASK] token"
    )
    parser.add_argument(
        "--random_token_prob", type=float, default=0.1, help="Probability of using random token"
    )

    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type")
    parser.add_argument("--devices", type=str, default="1", help="Number of devices")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Training precision")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--checkpoint_every_n_steps",
        type=int,
        default=0,
        help="Save checkpoint every N training steps (0 = disabled)",
    )
    parser.add_argument(
        "--checkpoint_every_n_epochs",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=0.25,
        help="How often to run validation within an epoch",
    )

    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="albatross-pretraining",
        help="W&B project name",
    )
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    print(f"Random seed set to: {args.seed}")

    current_date = datetime.now().strftime("%Y%m%d")
    lr_str = f"{args.lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    folder_name = f"albatross_{current_date}_lr{lr_str}_b{args.batch_size}"

    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Albatross IRES Pretraining")
    print("=" * 50)
    print(f"Training session folder: {folder_name}")
    print(f"Full output path: {output_dir}")
    print("=" * 50)

    sequences = load_sequences_from_file(args.data_path)

    if len(sequences) == 0:
        raise ValueError("No valid sequences found in the input file")

    n_val = max(1, int(len(sequences) * args.val_split)) if args.val_split > 0 else 0
    n_train = len(sequences) - n_val

    random.shuffle(sequences)
    train_sequences = sequences[:n_train]
    val_sequences = sequences[n_train:] if n_val > 0 else sequences[:1]

    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")

    alphabet = Alphabet()

    train_dataset = IRESDataset(
        train_sequences,
        alphabet,
        mask_ratio=args.mask_ratio,
        mask_token_prob=args.mask_token_prob,
        random_token_prob=args.random_token_prob,
        max_length=args.max_length,
    )

    val_dataset = IRESDataset(
        val_sequences,
        alphabet,
        mask_ratio=args.mask_ratio,
        mask_token_prob=args.mask_token_prob,
        random_token_prob=args.random_token_prob,
        max_length=args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = IRESPretrainingWrapper(
        lm_config=args.lm_config,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )

    if Path(args.pretrained_weights).exists():
        model.load_pretrained_weights(args.pretrained_weights)
    else:
        print(f"Warning: Pretrained weights not found at {args.pretrained_weights}")
        print("Training from scratch...")

    callbacks = []

    if args.checkpoint_every_n_steps > 0:
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="albatross-{epoch:02d}-{step:06d}",
            every_n_train_steps=args.checkpoint_every_n_steps,
            save_top_k=-1,
            save_last=True,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="albatross-{epoch:02d}-{step:06d}",
            every_n_epochs=args.checkpoint_every_n_epochs,
            save_top_k=-1,
            monitor="val/loss",
            mode="min",
            save_last=True,
        )
    callbacks.append(checkpoint_callback)

    loggers = []
    if args.wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            save_dir=output_dir,
            log_model=False,
        )
        loggers.append(wandb_logger)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    trainer_kwargs = {
        "accelerator": args.accelerator,
        "devices": args.devices,
        "max_epochs": args.max_epochs,
        "gradient_clip_val": args.gradient_clip_val,
        "precision": args.precision,
        "default_root_dir": output_dir,
        "callbacks": callbacks,
        "logger": loggers,
        "log_every_n_steps": 50,
        "val_check_interval": args.val_check_interval,
        "enable_checkpointing": True,
        "enable_progress_bar": True,
    }

    if args.max_steps is not None:
        trainer_kwargs["max_steps"] = args.max_steps

    trainer = pl.Trainer(**trainer_kwargs)

    print("Starting training...")
    if args.max_steps is not None:
        print(f"Total training steps: {args.max_steps}")
    else:
        print("Training steps: unlimited (using max_epochs only)")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Output directory: {output_dir}")

    trainer.fit(model, train_loader, val_loader)

    final_model_path = output_dir / "albatross_finetuned.pt"
    torch.save(model.model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
    print("Training completed!")


if __name__ == "__main__":
    main()
