import argparse
import datetime
import json
import logging
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def setup_logging(name: str, log_dir: str = "logs") -> logging.Logger:
    """Set up logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create formatters
    detail_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")

    # Create and setup file handler
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detail_formatter)

    # Create and setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with better defaults and validation."""
    parser = argparse.ArgumentParser(
        description="Graph Neural Network for Fraud Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add argument groups
    data_group = parser.add_argument_group("Data")
    model_group = parser.add_argument_group("Model")
    training_group = parser.add_argument_group("Training")
    system_group = parser.add_argument_group("System")

    # Data arguments
    data_group.add_argument(
        "--training-dir",
        type=str,
        default="./data/ieee_cis_clean/",
        help="Directory containing training data",
    )
    data_group.add_argument(
        "--nodes", type=str, default="features.csv", help="Node features file"
    )
    data_group.add_argument(
        "--target-ntype", type=str, default="TransactionID", help="Target node type"
    )
    data_group.add_argument(
        "--labels", type=str, default="tags.csv", help="Labels file"
    )

    # Model arguments
    model_group.add_argument(
        "--model-type",
        type=str,
        choices=["gat", "gcn", "gin"],
        default="gat",
        help="Type of GNN model",
    )
    model_group.add_argument(
        "--n-hidden", type=int, default=64, help="Hidden layer dimension"
    )
    model_group.add_argument(
        "--n-layers", type=int, default=3, help="Number of GNN layers"
    )
    model_group.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")

    # Training arguments
    training_group.add_argument(
        "--batch-size", type=int, default=8192, help="Batch size for training"
    )
    training_group.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    training_group.add_argument(
        "--weight-decay", type=float, default=1e-5, help="Weight decay"
    )
    training_group.add_argument(
        "--n-epochs", type=int, default=100, help="Number of epochs"
    )
    training_group.add_argument(
        "--patience", type=int, default=20, help="Patience for early stopping"
    )

    # System arguments
    system_group.add_argument(
        "--num-gpus", type=int, default=1, help="Number of GPUs to use"
    )
    system_group.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.num_gpus > torch.cuda.device_count():
        parser.error(
            f"Requested {args.num_gpus} GPUs but only {torch.cuda.device_count()} available"
        )

    # Create output directories
    args.model_dir = os.path.join(
        "models", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    args.output_dir = os.path.join(
        "outputs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    args.log_dir = os.path.join(
        "logs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    for dir_path in [args.model_dir, args.output_dir, args.log_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Save arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    return args


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, output_dir: str
) -> Dict[str, float]:
    """
    Compute and save classification metrics with plots.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        output_dir: Directory to save plots

    Returns:
        Dictionary of computed metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, output_dir)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plot_roc_curve(fpr, tpr, metrics["roc_auc"], output_dir)

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plot_pr_curve(precision, recall, metrics["avg_precision"], output_dir)

    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def plot_confusion_matrix(cm: np.ndarray, output_dir: str):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")

    # Add annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, output_dir: str):
    """Plot and save ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()


def plot_pr_curve(
    precision: np.ndarray, recall: np.ndarray, avg_precision: float, output_dir: str
):
    """Plot and save Precision-Recall curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        color="darkorange",
        lw=2,
        label=f"PR curve (AP = {avg_precision:.2f})",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "pr_curve.png"))
    plt.close()


class MetricTracker:
    """Track training metrics over time."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics = {
            "train": {"loss": [], "accuracy": [], "f1": []},
            "val": {"loss": [], "accuracy": [], "f1": []},
        }

    def update(self, split: str, metrics: Dict[str, float]):
        """Update metrics for given split."""
        for metric_name, value in metrics.items():
            self.metrics[split][metric_name].append(value)

    def plot_metrics(self):
        """Plot training curves."""
        metrics_to_plot = ["loss", "accuracy", "f1"]

        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))

            # Plot training metrics
            if metric in self.metrics["train"]:
                plt.plot(
                    self.metrics["train"][metric], label=f"Train {metric}", color="blue"
                )

            # Plot validation metrics
            if metric in self.metrics["val"]:
                plt.plot(
                    self.metrics["val"][metric], label=f"Val {metric}", color="orange"
                )

            plt.xlabel("Epoch")
            plt.ylabel(metric.capitalize())
            plt.title(f"Training and Validation {metric.capitalize()}")
            plt.legend()
            plt.grid(True)

            plt.savefig(os.path.join(self.output_dir, f"{metric}_curve.png"))
            plt.close()

    def save_metrics(self):
        """Save metrics to JSON file."""
        with open(os.path.join(self.output_dir, "training_metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_memory_stats() -> Dict[str, float]:
    """Get current GPU memory usage statistics."""
    if not torch.cuda.is_available():
        return {}

    stats = {
        "allocated": torch.cuda.memory_allocated() / 1024**2,  # MB
        "cached": torch.cuda.memory_reserved() / 1024**2,  # MB
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,
    }
    return stats


def log_gpu_memory(logger: logging.Logger):
    """Log current GPU memory usage."""
    memory_stats = get_memory_stats()
    if memory_stats:
        logger.info(
            "GPU Memory (MB): "
            f"Current={memory_stats['allocated']:.1f}, "
            f"Cached={memory_stats['cached']:.1f}, "
            f"Peak={memory_stats['max_allocated']:.1f}"
        )


class Timer:
    """Simple timer for code profiling."""

    def __init__(self, name: str, logger: logging.Logger):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        elapsed_time = self.start.elapsed_time(self.end)
        self.logger.info(f"{self.name} took {elapsed_time:.2f}ms")


# Initialize module logger
logger = setup_logging(__name__)
