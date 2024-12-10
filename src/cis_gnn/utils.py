import argparse
import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Graph Neural Network for Fraud Detection"
    )

    # Data parameters
    parser.add_argument(
        "--training-dir",
        type=str,
        default="./data/ieee_cis_clean/",
        help="Directory containing training data",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=f"./model/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        help="Directory to save model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save output",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        default="features.csv",
        help="Node features file",
    )
    parser.add_argument(
        "--target-ntype",
        type=str,
        default="TransactionID",
        help="Target node type",
    )
    parser.add_argument(
        "--edges",
        type=str,
        default="relation*",
        help="Edge list files pattern",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="tags.csv",
        help="Labels file",
    )
    parser.add_argument(
        "--new-accounts",
        type=str,
        default="test.csv",
        help="Test nodes file",
    )

    # GPU parameters
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (0 for CPU)",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Specific GPU ID to use",
    )

    # Model parameters
    parser.add_argument(
        "--compute-metrics",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=True,
        help="Compute evaluation metrics after training",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0,
        help="Threshold for making predictions",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="Optimizer to use",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Learning rate",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=700,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=16,
        help="Number of hidden units",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=3,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Weight decay for L2 regularization",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability",
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=360,
        help="Size of node embeddings",
    )

    args = parser.parse_known_args()[0]

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def get_logger(name):
    """Set up logger with standard format."""
    logger = logging.getLogger(name)
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger.setLevel(logging.INFO)
    return logger


def get_metrics(pred, pred_proba, labels, mask, out_dir):
    """
    Calculate various classification metrics.

    Args:
        pred: Binary predictions
        pred_proba: Prediction probabilities
        labels: True labels
        mask: Mask for test samples
        out_dir: Directory to save plots

    Returns:
        tuple: (accuracy, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix)
    """
    # Ensure everything is on CPU and in numpy format
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(pred_proba, torch.Tensor):
        pred_proba = pred_proba.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # Apply mask to get test samples
    mask = mask.astype(bool)
    labels = labels[mask]
    pred = pred[mask]
    pred_proba = pred_proba[mask]

    # Calculate basic metrics
    accuracy = (pred == labels).mean()

    # Calculate confusion matrix elements
    true_pos = ((pred == 1) & (labels == 1)).sum()
    false_pos = ((pred == 1) & (labels == 0)).sum()
    false_neg = ((pred == 0) & (labels == 1)).sum()
    true_neg = ((pred == 0) & (labels == 0)).sum()

    # Calculate precision and recall
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

    # Calculate F1 score
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Create confusion matrix
    confusion_matrix = np.array([[true_neg, false_pos], [false_neg, true_pos]])

    # Calculate ROC and PR curves
    fpr, tpr, _ = roc_curve(labels, pred_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(labels, pred_proba)

    # Calculate AUC scores
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall_curve, precision_curve)

    # Calculate average precision
    ap = average_precision_score(labels, pred_proba)

    # Save plots
    save_roc_curve(fpr, tpr, roc_auc, os.path.join(out_dir, "roc_curve.png"))
    save_pr_curve(
        precision_curve, recall_curve, pr_auc, ap, os.path.join(out_dir, "pr_curve.png")
    )

    return accuracy, f1, precision, recall, roc_auc, pr_auc, ap, confusion_matrix


def save_roc_curve(fpr, tpr, roc_auc, location):
    """Save ROC curve plot."""
    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")

    # Ensure directory exists
    os.makedirs(os.path.dirname(location), exist_ok=True)
    plt.savefig(location)
    plt.close()


def save_pr_curve(precision, recall, pr_auc, ap, location):
    """Save Precision-Recall curve plot."""
    plt.figure(figsize=(10, 8))
    plt.plot(
        recall,
        precision,
        color="darkorange",
        lw=2,
        label=f"PR curve (area = {pr_auc:.2f})",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP = {ap:.2f})")
    plt.legend(loc="lower right")

    # Ensure directory exists
    os.makedirs(os.path.dirname(location), exist_ok=True)
    plt.savefig(location)
    plt.close()


# Set up module logger
logger = get_logger(__name__)
