import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import random
import warnings

import numpy as np
from rich.logging import RichHandler
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import TqdmExperimentalWarning
import wandb

from src.config import ENTITY_NAME, MODEL_DIR
from src.yaku.exp1 import config as yaku_config
from src.yaku.exp1.training.model import DNN
from src.yaku.exp1.feature.yaku_encoder import YakuEncoder

torch.set_float32_matmul_precision("high")


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
        self.input_dir = data_dir / yaku_config.INPUT_NAME
        self.output_dir = data_dir / yaku_config.OUTPUT_NAME

        self.input_files = sorted(list(self.input_dir.glob("*.npy")))
        self.output_files = sorted(list(self.output_dir.glob("*.npy")))

        self.samples_per_file = yaku_config.TARGET_BATCH_SIZE
        self.total_samples = len(self.input_files) * self.samples_per_file

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.total_samples

    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        file_idx = idx // self.samples_per_file
        inner_idx = idx % self.samples_per_file

        inputs = np.load(self.input_files[file_idx], mmap_mode="r")
        outputs = np.load(self.output_files[file_idx], mmap_mode="r")

        return torch.from_numpy(inputs[inner_idx].copy()).float(), torch.from_numpy(outputs[inner_idx].copy()).float()


def setup_ddp(rank, world_size):
    """Set up Distributed Data Parallel (DDP) environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    """Clean up the distributed process group."""
    dist.destroy_process_group()


def _train_step(models, loader, loss_function, optimizer, device, num_yaku):
    """Execute a single training epoch."""
    for model in models:
        model.train()

    total_losses = np.zeros(num_yaku)
    total_correct = np.zeros(num_yaku)
    total_samples = 0

    for inputs, labels_all in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels_all = labels_all.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        batch_loss = 0

        for i, model in enumerate(models):
            outputs = model(inputs)
            labels = labels_all[:, i].unsqueeze(1)
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
def _valid_step(models, loader, loss_function, device, num_yaku):
    """Execute a single validation epoch."""
    for model in models:
        model.eval()

    total_losses = np.zeros(num_yaku)
    total_correct = np.zeros(num_yaku)
    total_samples = 0

    for inputs, labels_all in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels_all = labels_all.to(device, non_blocking=True)

        for i, model in enumerate(models):
            outputs = model(inputs)
            labels = labels_all[:, i].unsqueeze(1)
            loss = loss_function(outputs, labels)

            total_losses[i] += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct[i] += (preds == labels).sum().item()

        total_samples += inputs.size(0)

    return total_losses / total_samples, total_correct / total_samples


def train_yakus(rank, world_size, indices, yaku_names, parsed_args):
    """Train multiple yaku models simultaneously."""
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    num_yaku = len(indices)

    if rank == 0:
        wandb.init(
            entity=ENTITY_NAME,
            project=yaku_config.PROJECT_NAME,
            group=yaku_config.GROUP_NAME,
            name=parsed_args.name,
            config={**vars(yaku_config), "yaku_names": yaku_names},
        )

    train_dataset = YakuDataset(yaku_config.TRAIN_DIR)
    valid_dataset = YakuDataset(yaku_config.VALID_DIR)

    train_loader = DataLoader(
        train_dataset,
        batch_size=yaku_config.LEARNING_BATCH_SIZE,
        sampler=DistributedSampler(train_dataset, rank=rank, num_replicas=world_size),
        num_workers=os.cpu_count() // world_size,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=yaku_config.LEARNING_BATCH_SIZE,
        sampler=DistributedSampler(valid_dataset, rank=rank, num_replicas=world_size, shuffle=False),
        num_workers=os.cpu_count() // world_size,
        pin_memory=True,
    )

    models = nn.ModuleList(
        [
            DNN(
                input_dim=yaku_config.INPUT_DIM,
                hidden_layers=yaku_config.HIDDEN_LAYERS,
                output_dim=yaku_config.OUTPUT_DIM,
            ).to(device)
            for _ in range(num_yaku)
        ]
    )

    for i in range(num_yaku):
        try:
            models[i] = torch.compile(models[i])

        except Exception as e:
            if rank == 0:
                logging.warning("[%s] torch.compile failed: %s", yaku_names[i], e)

        models[i] = DDP(models[i], device_ids=[rank])

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(models.parameters(), lr=yaku_config.LEARNING_RATE)

    best_valid_losses = [float("inf")] * num_yaku
    early_stop_counters = [0] * num_yaku
    active_mask = [True] * num_yaku
    save_dirs = [MODEL_DIR / yaku_config.PROJECT_NAME / yaku_config.GROUP_NAME / name for name in yaku_names]

    if rank == 0:
        for save_dir in save_dirs:
            save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(yaku_config.MAX_EPOCHS):
        train_loader.sampler.set_epoch(epoch)

        train_losses, train_accs = _train_step(models, train_loader, loss_function, optimizer, device, num_yaku)
        valid_losses, valid_accs = _valid_step(models, valid_loader, loss_function, device, num_yaku)

        combined = np.stack([train_losses, train_accs, valid_losses, valid_accs])
        metrics = torch.tensor(combined, device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics /= world_size
        sync_train_losses, sync_train_accs, sync_valid_losses, sync_valid_accs = metrics.cpu().numpy()

        if rank == 0:
            log_dict = {"epoch": epoch + 1}

            for i in range(num_yaku):
                yaku_name = yaku_names[i]
                logging.info(
                    "[%s] Epoch %d: Train Loss %.4f | Valid Loss %.4f, Train Acc %.4f | Valid Acc %.4f",
                    yaku_name,
                    epoch + 1,
                    sync_train_losses[i],
                    sync_valid_losses[i],
                    sync_train_accs[i],
                    sync_valid_accs[i],
                )

                log_dict.update(
                    {
                        f"{yaku_name}/train_loss": sync_train_losses[i],
                        f"{yaku_name}/train_acc": sync_train_accs[i],
                        f"{yaku_name}/valid_loss": sync_valid_losses[i],
                        f"{yaku_name}/valid_acc": sync_valid_accs[i],
                    }
                )

                if active_mask[i]:
                    log_dict[f"{yaku_name}/best_valid_loss"] = best_valid_losses[i]

                    if sync_valid_losses[i] < best_valid_losses[i]:
                        best_valid_losses[i] = sync_valid_losses[i]
                        log_dict[f"{yaku_name}/best_valid_loss"] = best_valid_losses[i]
                        early_stop_counters[i] = 0
                        torch.save(models[i].module.state_dict(), save_dirs[i] / "best_valid_loss_model.pth")

                    else:
                        early_stop_counters[i] += 1

                        if early_stop_counters[i] >= yaku_config.EARLY_STOPPING_PATIENCE:
                            logging.info("[%s] Early Stopping reached.", yaku_name)
                            active_mask[i] = False
                            log_dict[f"{yaku_name}/early_stop_epoch"] = epoch + 1

            wandb.log(log_dict)

            if not any(active_mask):
                logging.info("All models reached early stopping.")
                break

        dist.barrier()

    if rank == 0:
        wandb.finish()


def main(rank, world_size, parsed_args):
    """Main function."""
    setup_ddp(rank, world_size)

    random.seed(yaku_config.SEED)
    np.random.seed(yaku_config.SEED)
    torch.manual_seed(yaku_config.SEED)

    yaku_encoder = YakuEncoder()
    indices = [int(index.strip()) for index in parsed_args.yaku_indices.split(",")]
    yaku_names = [yaku_encoder.get_name(index) for index in indices]

    if rank == 0:
        logging.info("Starting training for Yakus: %s", ", ".join(yaku_names))

    train_yakus(rank, world_size, indices, yaku_names, parsed_args)

    cleanup_ddp()


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaku_indices", type=str, default="0,1,2,3,4,5,6,7,8")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--name", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    num_gpus = len(args.gpus.split(","))

    mp.spawn(main, args=(num_gpus, args), nprocs=num_gpus, join=True)
