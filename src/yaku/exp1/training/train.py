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
from src.yaku.exp1.training.model import DNN


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

    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        file_idx = np.searchsorted(self.file_offsets, idx, side="right") - 1
        inner_idx = idx - self.file_offsets[file_idx]

        inputs = np.load(self.input_files[file_idx], mmap_mode="r")
        outputs = np.load(self.output_files[file_idx], mmap_mode="r")

        return torch.from_numpy(inputs[inner_idx].copy()).float(), torch.from_numpy(outputs[inner_idx].copy()).float()


def _train_step(models, indices, loader, loss_function, optimizer, device, num_yaku, epoch):
    """Execute a single training epoch with progress bar."""
    for model in models:
        model.train()

    total_losses = np.zeros(num_yaku)
    total_correct = np.zeros(num_yaku)
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for inputs, labels_all in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels_all = labels_all.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        batch_loss = 0

        for i, model in enumerate(models):
            outputs = model(inputs)
            labels = labels_all[:, indices[i]].unsqueeze(1)
            loss = loss_function(outputs, labels)

            batch_loss += loss

            total_losses[i] += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct[i] += (preds == labels).sum().item()

        batch_loss.backward()
        optimizer.step()
        total_samples += inputs.size(0)

    return total_losses / total_samples, total_correct / total_samples


@torch.no_grad()
def _valid_step(models, indices, loader, loss_function, device, num_yaku, epoch):
    """Execute a single validation epoch with progress bar."""
    for model in models:
        model.eval()

    total_losses = np.zeros(num_yaku)
    total_correct = np.zeros(num_yaku)
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Valid]", leave=False)

    for inputs, labels_all in pbar:
        inputs = inputs.to(device, non_blocking=True)
        labels_all = labels_all.to(device, non_blocking=True)

        for i, model in enumerate(models):
            outputs = model(inputs)
            labels = labels_all[:, indices[i]].unsqueeze(1)
            loss = loss_function(outputs, labels)

            total_losses[i] += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct[i] += (preds == labels).sum().item()

        total_samples += inputs.size(0)

    return total_losses / total_samples, total_correct / total_samples


def train_yakus(indices, yaku_names, parsed_args, device):
    """Train multiple yaku models simultaneously."""
    num_yaku = len(indices)

    use_wandb = not parsed_args.no_wandb

    if use_wandb:
        logging.info("Initializing WandB...")
        clean_config = {}

        for k, v in vars(exp1_config).items():
            if k.isupper():
                if isinstance(v, pathlib.Path):
                    clean_config[k] = str(v)

                else:
                    clean_config[k] = v

        clean_config["yaku_names"] = yaku_names

        wandb.init(
            project=common_config.PROJECT_NAME,
            group=exp1_config.GROUP_NAME,
            name=parsed_args.name,
            config=clean_config,
        )

    logging.info("Preparing Datasets and DataLoaders...")
    train_dataset = YakuDataset(exp1_config.TRAIN_DIR)
    valid_dataset = YakuDataset(exp1_config.VALID_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=exp1_config.LEARNING_BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=exp1_config.LEARNING_BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    logging.info("Initializing %d models on device: %s", num_yaku, device)
    models = nn.ModuleList(
        [
            DNN(
                input_dim=exp1_config.INPUT_DIM,
                hidden_layers=exp1_config.HIDDEN_LAYERS,
                output_dim=exp1_config.OUTPUT_DIM,
            ).to(device)
            for _ in range(num_yaku)
        ]
    )

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(models.parameters(), lr=exp1_config.LEARNING_RATE)

    best_valid_losses = [float("inf")] * num_yaku
    early_stop_counters = [0] * num_yaku
    active_mask = [True] * num_yaku
    save_dirs = [MODEL_DIR / common_config.PROJECT_NAME / exp1_config.GROUP_NAME / name for name in yaku_names]

    for save_dir in save_dirs:
        save_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Starting training loop (Total Epochs: %d)...", exp1_config.MAX_EPOCHS)

    for epoch in range(exp1_config.MAX_EPOCHS):
        logging.info("Epoch %d/%d: Training phase...", epoch + 1, exp1_config.MAX_EPOCHS)
        train_losses, train_accs = _train_step(
            models, indices, train_loader, loss_function, optimizer, device, num_yaku, epoch + 1
        )

        logging.info("Epoch %d/%d: Validation phase...", epoch + 1, exp1_config.MAX_EPOCHS)
        valid_losses, valid_accs = _valid_step(
            models, indices, valid_loader, loss_function, device, num_yaku, epoch + 1
        )

        log_dict = {"epoch": epoch + 1}

        for i in range(num_yaku):
            yaku_name = yaku_names[i]
            logging.info(
                "[%s] Epoch %d: Train Loss %.4f | Valid Loss %.4f, Train Acc %.4f | Valid Acc %.4f",
                yaku_name,
                epoch + 1,
                train_losses[i],
                valid_losses[i],
                train_accs[i],
                valid_accs[i],
            )

            if use_wandb:
                log_dict.update(
                    {
                        f"{yaku_name}/train_loss": train_losses[i],
                        f"{yaku_name}/train_acc": train_accs[i],
                        f"{yaku_name}/valid_loss": valid_losses[i],
                        f"{yaku_name}/valid_acc": valid_accs[i],
                    }
                )

            if active_mask[i]:
                if valid_losses[i] < best_valid_losses[i]:
                    best_valid_losses[i] = valid_losses[i]
                    early_stop_counters[i] = 0
                    torch.save(models[i].state_dict(), save_dirs[i] / "best_valid_loss_model.pth")

                else:
                    early_stop_counters[i] += 1

                    if early_stop_counters[i] >= exp1_config.EARLY_STOPPING_PATIENCE:
                        logging.info("[%s] Early Stopping reached.", yaku_name)
                        active_mask[i] = False

                        if use_wandb:
                            log_dict[f"{yaku_name}/early_stop_epoch"] = epoch + 1

                if use_wandb:
                    log_dict[f"{yaku_name}/best_valid_loss"] = best_valid_losses[i]

        if use_wandb:
            wandb.log(log_dict)

        if not any(active_mask):
            logging.info("All models reached early stopping.")
            break

    if use_wandb:
        wandb.finish()


def main(parsed_args):
    """Main function for training."""
    random.seed(common_config.SEED)
    np.random.seed(common_config.SEED)
    torch.manual_seed(common_config.SEED)

    device = torch.device(f"cuda:{parsed_args.gpu}" if torch.cuda.is_available() else "cpu")

    yaku_encoder = YakuEncoder()
    indices = [int(index.strip()) for index in parsed_args.yaku_indices.split(",")]
    yaku_names = [yaku_encoder.index_to_name[index] for index in indices]

    logging.info("Starting training for Yakus: %s", ", ".join(yaku_names))

    train_yakus(indices, yaku_names, parsed_args, device)


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaku_indices",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32",
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--name", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    args = parser.parse_args()

    main(args)
