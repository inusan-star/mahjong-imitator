import argparse
import logging
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from rich.logging import RichHandler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from tqdm import TqdmExperimentalWarning

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

        for file_path in self.input_files:
            size = np.load(file_path, mmap_mode="r").shape[0]
            self.file_offsets.append(self.total_samples)
            self.total_samples += size

        self.file_offsets = np.array(self.file_offsets)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.total_samples

    def __getitem__(self, idx: int):
        """Get a single sample from the dataset."""
        file_index = np.searchsorted(self.file_offsets, idx, side="right") - 1
        inner_index = idx - self.file_offsets[file_index]

        inputs = np.load(self.input_files[file_index], mmap_mode="r")
        outputs = np.load(self.output_files[file_index], mmap_mode="r")

        return (
            torch.from_numpy(inputs[inner_index].copy()).float(),
            torch.from_numpy(outputs[inner_index].copy()).float(),
        )


@torch.no_grad()
def evaluate_model(indices: list, yaku_names: list, loader: DataLoader, device: torch.device) -> dict:
    """Evaluate single multi-task model and return metrics."""
    num_yaku = len(indices)

    model_path = MODEL_DIR / common_config.PROJECT_NAME / exp2_config.GROUP_NAME / "best_model.pth"
    model = DNN(
        input_dim=exp2_config.INPUT_DIM,
        hidden_layers=exp2_config.HIDDEN_LAYERS,
        output_dim=exp2_config.OUTPUT_DIM,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loss_function = nn.BCEWithLogitsLoss(reduction="none")

    all_logits = [[] for _ in range(num_yaku)]
    all_labels = [[] for _ in range(num_yaku)]
    yaku_losses = np.zeros(num_yaku)
    total_samples = 0

    progress_bar = tqdm(loader, desc="Evaluating", dynamic_ncols=True)

    for inputs, labels_all in progress_bar:
        inputs = inputs.to(device, non_blocking=True)
        labels_all = labels_all.to(device, non_blocking=True)
        total_samples += inputs.size(0)

        outputs_all = model(inputs)
        losses_all = loss_function(outputs_all, labels_all)

        yaku_losses += losses_all.sum(dim=0).cpu().numpy()

        for i in range(num_yaku):
            all_logits[i].extend(outputs_all[:, i].cpu().numpy().flatten())
            all_labels[i].extend(labels_all[:, i].cpu().numpy().flatten())

    basic_metrics = []
    curve_metrics = []

    for i in range(num_yaku):
        logits = np.array(all_logits[i])
        labels = np.array(all_labels[i])
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > 0.5).astype(float)

        precision, recall, f1_score, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        accuracy = accuracy_score(labels, preds)

        roc_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else np.nan
        pr_auc = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else np.nan

        basic_metrics.append(
            {
                "yaku_name": yaku_names[i],
                "loss": yaku_losses[i] / total_samples,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            }
        )

        curve_metrics.append(
            {
                "yaku_name": yaku_names[i],
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

    return {"basic": basic_metrics, "curve": curve_metrics}


def main(parsed_args: argparse.Namespace):
    """Main function for evaluation."""
    device = torch.device(f"cuda:{parsed_args.gpu}" if torch.cuda.is_available() else "cpu")

    yaku_encoder = YakuEncoder()
    yaku_names = yaku_encoder.labels
    indices = list(range(len(yaku_names)))

    logging.info("Preparing Dataset and DataLoader...")
    test_dataset = YakuDataset(exp1_config.VALID_DIR)
    test_loader = DataLoader(
        test_dataset,
        batch_size=exp2_config.LEARNING_BATCH_SIZE,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    logging.info("Starting evaluation for %d Yakus (Multi-task)...", len(yaku_names))
    evaluation_results = evaluate_model(indices, yaku_names, test_loader, device)

    output_directory = Path("results") / "exp2"
    output_directory.mkdir(parents=True, exist_ok=True)

    df_basic = pd.DataFrame(evaluation_results["basic"])
    df_curve = pd.DataFrame(evaluation_results["curve"])

    df_basic.to_csv(output_directory / "metrics_basic.csv", index=False, float_format="%.5f")
    df_curve.to_csv(output_directory / "metrics_curves.csv", index=False, float_format="%.5f")

    logging.info("Evaluation results saved to: %s", output_directory)

    mean_basic = df_basic.drop(columns=["yaku_name"]).mean()
    mean_curve = df_curve.drop(columns=["yaku_name"]).mean()

    pd.DataFrame([mean_basic]).to_csv(output_directory / "mean_basic.csv", index=False, float_format="%.5f")
    pd.DataFrame([mean_curve]).to_csv(output_directory / "mean_curves.csv", index=False, float_format="%.5f")

    logging.info("Average metrics saved.")


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    main(args)
