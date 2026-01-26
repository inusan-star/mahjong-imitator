import argparse
from datetime import datetime
import logging
import os
import pathlib
from pathlib import Path
import random
import warnings

import numpy as np
from rich.logging import RichHandler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from tqdm import TqdmExperimentalWarning
import wandb

from src.config import MODEL_DIR
import src.yaku.common.config as common_config
from src.yaku.common.yaku_encoder import YakuEncoder
import src.yaku.exp1.config as exp1_config
import src.yaku.exp2.config as exp2_config
from src.yaku.exp2.training.model import DNN


def setup_logging():
    """Set up logging configuration."""
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                log_time_format="[%Y-%m-%d %H:%M:%S]",
            )
        ],
    )


class YakuDataset(Dataset):
    """Dataset for loading yaku training data."""

    def __init__(self, data_dir: Path):
        """Initialize the dataset with input/output directories."""
        self.input_dir = data_dir / exp1_config.INPUT_FILENAME_PREFIX
        self.output_dir = data_dir / exp1_config.OUTPUT_FILENAME_PREFIX

        self.input_files = sorted(list(self.input_dir.glob("*.npy")))
        self.output_files = sorted(list(self.output_dir.glob("*.npy")))

        self.file_offsets = []
        self.total_samples = 0

        for f in self.input_files:
            size = np.load(f, mmap_mode="r").shape[0]
            self.file_offsets.append(self.total_samples)
            self.total_samples += size

        self.file_offsets = np.array(self.file_offsets)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.total_samples

    def __getitem__(self, idx: int):
        """Get a single sample from the dataset."""
        file_idx = np.searchsorted(self.file_offsets, idx, side="right") - 1
        inner_idx = idx - self.file_offsets[file_idx]

        inputs = np.load(self.input_files[file_idx], mmap_mode="r")
        outputs = np.load(self.output_files[file_idx], mmap_mode="r")

        return torch.from_numpy(inputs[inner_idx].copy()).float(), torch.from_numpy(outputs[inner_idx].copy()).float()


def _train_step(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_yaku: int,
    epoch: int,
):
    """Execute a single training epoch."""
    model.train()

    total_losses = np.zeros(num_yaku)
    total_correct = np.zeros(num_yaku)
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    # 役ごとの統計を計算するための損失関数（リダクションなし）
    individual_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)

        # 全体の最適化用ロス
        batch_loss = loss_function(outputs, labels)
        batch_loss.backward()
        optimizer.step()

        # 統計の計算
        with torch.no_grad():
            losses = individual_loss_fn(outputs, labels)
            total_losses += losses.sum(dim=0).cpu().numpy()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == labels).sum(dim=0).cpu().numpy()
            total_samples += inputs.size(0)

    return total_losses / total_samples, total_correct / total_samples


@torch.no_grad()
def _valid_step(
    model: nn.Module, loader: DataLoader, loss_function: nn.Module, device: torch.device, num_yaku: int, epoch: int
):
    """Execute a single validation epoch."""
    model.eval()

    total_losses = np.zeros(num_yaku)
    total_correct = np.zeros(num_yaku)
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Valid]", leave=False)
    individual_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    for inputs, labels in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        losses = individual_loss_fn(outputs, labels)

        total_losses += losses.sum(dim=0).cpu().numpy()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        total_correct += (preds == labels).sum(dim=0).cpu().numpy()
        total_samples += inputs.size(0)

    return total_losses / total_samples, total_correct / total_samples


def train_yakus(yaku_names: list, parsed_args: argparse.Namespace, device: torch.device):
    """Train a single multi-task model for all yakus."""
    num_yaku = len(yaku_names)
    use_wandb = not parsed_args.no_wandb

    if use_wandb:
        logging.info("Initializing WandB...")
        clean_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(exp2_config).items() if k.isupper()}
        clean_config["yaku_names"] = yaku_names

        wandb.init(
            project=common_config.PROJECT_NAME,
            group=exp2_config.GROUP_NAME,
            name=parsed_args.name,
            config=clean_config,
        )

    logging.info("Preparing Datasets and DataLoaders (using Exp1 data)...")
    train_dataset = YakuDataset(exp1_config.TRAIN_DIR)
    valid_dataset = YakuDataset(exp1_config.VALID_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=exp2_config.LEARNING_BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=exp2_config.LEARNING_BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    logging.info("Initializing Multi-task DNN on device: %s", device)
    model = DNN(
        input_dim=exp2_config.INPUT_DIM,
        hidden_layers=exp2_config.HIDDEN_LAYERS,
        output_dim=num_yaku,
    ).to(device)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=exp2_config.LEARNING_RATE)

    best_mean_valid_loss = float("inf")
    early_stop_counter = 0
    save_dir = MODEL_DIR / common_config.PROJECT_NAME / exp2_config.GROUP_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Starting training loop (Total Epochs: %d)...", exp2_config.MAX_EPOCHS)

    for epoch in range(exp2_config.MAX_EPOCHS):
        train_losses, train_accs = _train_step(
            model, train_loader, loss_function, optimizer, device, num_yaku, epoch + 1
        )
        valid_losses, valid_accs = _valid_step(model, valid_loader, loss_function, device, num_yaku, epoch + 1)

        mean_valid_loss = np.mean(valid_losses)
        log_dict = {"epoch": epoch + 1, "mean_valid_loss": mean_valid_loss}

        for i in range(num_yaku):
            yaku_name = yaku_names[i]
            if use_wandb:
                log_dict.update(
                    {
                        f"{yaku_name}/train_loss": train_losses[i],
                        f"{yaku_name}/valid_loss": valid_losses[i],
                        f"{yaku_name}/valid_acc": valid_accs[i],
                    }
                )

        if use_wandb:
            wandb.log(log_dict)

        logging.info("Epoch %d: Mean Valid Loss %.4f", epoch + 1, mean_valid_loss)

        if mean_valid_loss < best_mean_valid_loss:
            best_mean_valid_loss = mean_valid_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            logging.info("Best model saved at epoch %d", epoch + 1)
        else:
            early_stop_counter += 1
            if early_stop_counter >= exp2_config.EARLY_STOPPING_PATIENCE:
                logging.info("Early Stopping reached (Mean Valid Loss).")
                break

    if use_wandb:
        wandb.finish()


def main(parsed_args: argparse.Namespace):
    """Main function for multi-task training."""
    random.seed(common_config.SEED)
    np.random.seed(common_config.SEED)
    torch.manual_seed(common_config.SEED)

    device = torch.device(f"cuda:{parsed_args.gpu}" if torch.cuda.is_available() else "cpu")

    yaku_encoder = YakuEncoder()
    yaku_names = yaku_encoder.labels

    logging.info("Starting Multi-task training for %d Yakus", len(yaku_names))
    train_yakus(yaku_names, parsed_args, device)


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--name", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    args = parser.parse_args()

    main(args)
