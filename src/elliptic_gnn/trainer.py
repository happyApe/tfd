import os
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import EllipticDataset
from models import GAT, GCN, GIN
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.tensorboard import SummaryWriter  # type: ignore


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataset: EllipticDataset,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 0.01,
        weight_decay: float = 1e-5,
        tensorboard_dir: Optional[str] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: Neural network model to train
            dataset: Dataset to train on
            device: Device to use for training ('cuda' or 'cpu')
            lr: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            tensorboard_dir: Directory for tensorboard logs (optional)
        """
        self.device = device
        self.dataset = dataset.pyg_dataset().to(device)
        self.model = model.double().to(device)

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )

        if tensorboard_dir:
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tensorboard = SummaryWriter(tensorboard_dir)
        else:
            self.tensorboard = None

        self.metrics_history = {
            "train": {
                "loss": [],
                "accuracy": [],
                "f1_micro": [],
                "f1_macro": [],
                "recall": [],
                "precision": [],
                "confusion_matrix": [],
            },
            "valid": {
                "loss": [],
                "accuracy": [],
                "f1_micro": [],
                "f1_macro": [],
                "recall": [],
                "precision": [],
                "confusion_matrix": [],
            },
        }

    def compute_metrics(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        loss: Optional[float] = None,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            preds: Model predictions
            labels: True labels
            loss: Optional loss value
            threshold: Classification threshold

        Returns:
            Dictionary containing computed metrics
        """
        preds_binary = (preds > threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(labels, preds_binary),
            "f1_micro": f1_score(labels, preds_binary, average="micro"),
            "f1_macro": f1_score(labels, preds_binary, average="macro"),
            "recall": recall_score(labels, preds_binary),
            "precision": precision_score(labels, preds_binary, zero_division=1),
        }

        if loss is not None:
            metrics["loss"] = loss

        return metrics

    def train_epoch(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Train for one epoch.

        Returns:
            Tuple of dictionaries containing training and validation metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(self.dataset)
        outputs = outputs.reshape((self.dataset.x.shape[0]))

        # Calculate training loss
        train_loss = self.criterion(
            outputs[self.dataset.train_idx], self.dataset.y[self.dataset.train_idx]
        )

        # Backward pass
        train_loss.backward()
        self.optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            # Training metrics
            train_preds = outputs[self.dataset.train_idx].cpu().numpy()
            train_labels = self.dataset.y[self.dataset.train_idx].cpu().numpy()
            train_metrics = self.compute_metrics(
                train_preds, train_labels, loss=train_loss.item()
            )

            # Validation metrics
            valid_preds = outputs[self.dataset.valid_idx].cpu().numpy()
            valid_labels = self.dataset.y[self.dataset.valid_idx].cpu().numpy()
            valid_loss = self.criterion(
                outputs[self.dataset.valid_idx], self.dataset.y[self.dataset.valid_idx]
            )
            valid_metrics = self.compute_metrics(
                valid_preds, valid_labels, loss=valid_loss.item()
            )

        return train_metrics, valid_metrics

    def train(
        self,
        num_epochs: int = 100,
        print_freq: int = 10,
        early_stopping_patience: int = 20,
    ) -> float:
        """
        Train the model.

        Args:
            num_epochs: Number of training epochs
            print_freq: How often to print progress
            early_stopping_patience: Number of epochs to wait for improvement before stopping

        Returns:
            Best validation F1 score achieved
        """
        best_valid_f1 = 0
        best_model_state = None
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            train_metrics, valid_metrics = self.train_epoch()

            # Update learning rate scheduler
            self.scheduler.step(valid_metrics["loss"])

            # Update metrics history
            for mode, metrics in [("train", train_metrics), ("valid", valid_metrics)]:
                for metric_name, value in metrics.items():
                    self.metrics_history[mode][metric_name].append(value)
                    if self.tensorboard:
                        self.tensorboard.add_scalar(
                            f"{mode}/{metric_name}", value, epoch
                        )

            # Save best model
            if valid_metrics["f1_macro"] > best_valid_f1:
                best_valid_f1 = valid_metrics["f1_macro"]
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            if epoch % print_freq == 0 or epoch == 1:
                print(
                    f"Epoch {epoch}/{num_epochs}:\n"
                    f"Train metrics: {train_metrics}\n"
                    f"Valid metrics: {valid_metrics}\n"
                )

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nRestored best model with validation F1: {best_valid_f1:.4f}")

        return best_valid_f1

    def test(
        self,
        dataset: Optional[Any] = None,
        labeled_only: bool = False,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Test the model.

        Args:
            dataset: Optional dataset to test on (uses training dataset if None)
            labeled_only: Whether to return predictions for labeled data only
            threshold: Classification threshold

        Returns:
            Tuple of (raw predictions, binary predictions)
        """
        dataset = dataset or self.dataset

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(dataset)
            outputs = outputs.reshape((dataset.x.shape[0]))

            if labeled_only:
                preds = outputs.cpu().numpy()
            else:
                preds = outputs[dataset.test_idx].cpu().numpy()

            pred_labels = (preds > threshold).astype(int)

        return preds, pred_labels

    def save(self, save_path: str):
        """
        Save the model weights.

        Args:
            save_path: Path to save the model
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "metrics_history": self.metrics_history,
            },
            save_path,
        )
        print(f"Model saved to {save_path}")

    def load(self, load_path: str):
        """
        Load model weights and training state.

        Args:
            load_path: Path to load the model from
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.metrics_history = checkpoint["metrics_history"]
        print(f"Model loaded from {load_path}")

    def visualize(
        self,
        dataset: EllipticDataset,
        time_step: int,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 16),
    ):
        """
        Visualize the network for a specific time step.

        Args:
            dataset: Dataset to visualize
            time_step: Time step to visualize
            save_path: Optional path to save the visualization
            figsize: Figure size for the plot
        """
        pred_scores, pred_labels = self.test(
            dataset.pyg_dataset().to(self.device), labeled_only=True
        )

        # Get nodes for the specific time step
        node_list = dataset.merged_df.index[
            dataset.merged_df.loc[:, 1] == time_step
        ].tolist()

        # Create edge tuples
        edge_tuples = []
        for row in dataset.edge_index.view(-1, 2).cpu().numpy():
            if (row[0] in node_list) or (row[1] in node_list):
                edge_tuples.append(tuple(row))

        # Assign colors based on true labels and predictions
        node_colors = []
        for node_id in node_list:
            if node_id in dataset.illicit_ids:
                color = "red"  # True fraud
            elif node_id in dataset.licit_ids:
                color = "green"  # True legitimate
            else:
                color = (
                    "orange" if pred_labels[node_id] else "blue"
                )  # Predicted fraud/legitimate
            node_colors.append(color)

        # Create and plot the graph
        G = nx.Graph()
        G.add_edges_from(edge_tuples)

        plt.figure(figsize=figsize)
        plt.title(f"Transaction Network - Time Period: {time_step}")

        # Draw the network
        nx.draw_networkx(
            G,
            nodelist=node_list,
            node_color=node_colors,
            node_size=6,
            with_labels=False,
            alpha=0.7,
            edge_color="gray",
            width=0.5,
        )

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="True Fraud",
                markerfacecolor="red",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="True Legitimate",
                markerfacecolor="green",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Predicted Fraud",
                markerfacecolor="orange",
                markersize=10,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Predicted Legitimate",
                markerfacecolor="blue",
                markersize=10,
            ),
        ]
        plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.1, 1))

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"Graph visualization saved to {save_path}")
        else:
            plt.show()
