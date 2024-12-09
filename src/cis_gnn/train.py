import copy
import os
import pickle
import sys
import time
from typing import Dict, Optional, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from graph_utils import construct_graph, get_edgelists, get_labels
from model import HeteroRGCN
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from utils import get_logger, get_metrics, parse_args

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: HeteroRGCN,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        device: torch.device,
        model_dir: str,
        mixed_precision: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_dir = model_dir
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None

        self.best_model = None
        self.best_val_score = float("-inf")
        self.best_epoch = 0

    def train_epoch(
        self,
        train_g: dgl.DGLGraph,
        features: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()

        with autocast(enabled=self.mixed_precision):
            # Forward pass
            logits = self.model(train_g, features)
            loss = self.model.get_loss(logits[valid_mask], labels[valid_mask])

        # Backward pass with gradient scaling
        self.optimizer.zero_grad()
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        self.scheduler.step()

        # Calculate metrics
        with torch.no_grad():
            f1_score = self.evaluate(train_g, features, labels, valid_mask)

        return loss.item(), f1_score

    @torch.no_grad()
    def evaluate(
        self,
        g: dgl.DGLGraph,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """Evaluate model."""
        self.model.eval()
        logits = self.model(g, features)
        preds = torch.argmax(logits[mask], axis=1)
        return self.compute_f1_score(labels[mask].cpu(), preds.cpu())

    def train(
        self,
        train_g: dgl.DGLGraph,
        features: torch.Tensor,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor,
        n_epochs: int,
        patience: int = 20,
    ):
        """Full training loop with early stopping."""
        patience_counter = 0
        train_times = []

        for epoch in range(n_epochs):
            tic = time.time()

            # Training step
            loss, train_f1 = self.train_epoch(train_g, features, labels, train_mask)

            # Validation step
            val_f1 = self.evaluate(train_g, features, labels, val_mask)

            # Update best model
            if val_f1 > self.best_val_score:
                self.best_val_score = val_f1
                self.best_model = copy.deepcopy(self.model)
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            # Timing and logging
            epoch_time = time.time() - tic
            train_times.append(epoch_time)

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch:05d} | Time(s) {np.mean(train_times):.4f} | "
                    f"Loss {loss:.4f} | Train F1 {train_f1:.4f} | Val F1 {val_f1:.4f}"
                )

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Final evaluation
        self.model = self.best_model
        test_f1 = self.evaluate(train_g, features, labels, test_mask)
        logger.info(f"Test F1: {test_f1:.4f}")

        return test_f1

    @staticmethod
    def compute_f1_score(labels: torch.Tensor, preds: torch.Tensor) -> float:
        """Compute F1 score."""
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return 2 * (precision * recall) / (precision + recall + 1e-8)

    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
        }

        path = os.path.join(self.model_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint["epoch"], checkpoint["metrics"]


def main():
    args = parse_args()
    logger.info(f"Running with args: {args}")

    # Set up device and environment
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.num_gpus > 0 else "cpu"
    )
    torch.backends.cudnn.benchmark = True

    # Construct graph and prepare data
    args.edges = get_edgelists("relation*", args.training_dir)
    g, features, target_id_to_node, id_to_node = construct_graph(
        args.training_dir, args.edges, args.nodes, args.target_ntype
    )

    # Feature preprocessing
    features = torch.from_numpy(features).float()
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True) + 1e-6
    features = (features - mean) / std

    # Get labels and create masks
    n_nodes = g.number_of_nodes("target")
    labels, train_mask, test_mask = get_labels(
        target_id_to_node,
        n_nodes,
        args.target_ntype,
        os.path.join(args.training_dir, args.labels),
        os.path.join(args.training_dir, args.new_accounts),
    )

    # Split train into train/val
    train_indices = torch.where(train_mask)[0]
    val_size = int(0.2 * len(train_indices))
    val_indices = train_indices[:val_size]
    train_indices = train_indices[val_size:]

    val_mask = torch.zeros_like(train_mask)
    val_mask[val_indices] = 1
    train_mask[val_indices] = 0

    # Convert to tensors and move to device
    features = features.to(device)
    labels = torch.from_numpy(labels).long().to(device)
    train_mask = torch.from_numpy(train_mask).to(device)
    val_mask = val_mask.to(device)
    test_mask = torch.from_numpy(test_mask).to(device)
    g = g.to(device)

    # Initialize model
    in_feats = features.shape[1]
    ntype_dict = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}

    model = HeteroRGCN(
        ntype_dict=ntype_dict,
        etypes=g.etypes,
        in_size=in_feats,
        hidden_size=args.n_hidden,
        out_size=2,  # binary classification
        n_layers=args.n_layers,
        embedding_size=args.embedding_size,
        num_heads=4,
        dropout=args.dropout,
        use_attention=True,
    ).to(device)

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True
    )

    # OneCycle scheduler for better training dynamics
    steps_per_epoch = 1  # since we're doing full-batch training
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.n_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=10000.0,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_dir=args.model_dir,
        mixed_precision=True,
    )

    # Training loop
    logger.info("Starting training...")
    os.makedirs(args.output_dir, exist_ok=True)

    test_f1 = trainer.train(
        train_g=g,
        features=features,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        n_epochs=args.n_epochs,
        patience=20,
    )

    # Final evaluation and predictions
    trainer.model.eval()
    with torch.no_grad():
        logits = trainer.model(g, features)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

    # Save predictions
    test_indices = torch.where(test_mask)[0].cpu()
    predictions_df = pd.DataFrame(
        {
            "node_idx": test_indices.numpy(),
            "prediction": preds[test_mask].cpu().numpy(),
            "probability": probs[test_mask, 1].cpu().numpy(),
        }
    )
    predictions_df.to_csv(os.path.join(args.output_dir, "predictions.csv"), index=False)

    # Save model and metadata
    save_model_and_metadata(
        g=g,
        model=trainer.model,
        model_dir=args.model_dir,
        id_to_node=id_to_node,
        feature_stats={"mean": mean, "std": std},
        metrics={"test_f1": test_f1},
    )

    logger.info("Training completed successfully!")


def save_model_and_metadata(g, model, model_dir, id_to_node, feature_stats, metrics):
    """Save model, embeddings, and associated metadata."""
    os.makedirs(model_dir, exist_ok=True)

    # Save model state
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))

    # Save metadata
    metadata = {
        "etypes": g.canonical_etypes,
        "ntype_cnt": {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes},
        "feature_stats": feature_stats,
        "metrics": metrics,
    }

    with open(os.path.join(model_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    # Save node embeddings for each type
    for ntype, mapping in id_to_node.items():
        if ntype == "target":
            continue

        # Get the learned embeddings
        embeddings = model.embed[ntype].detach().cpu().numpy()

        # Create DataFrame with node information
        embedding_df = pd.DataFrame(
            embeddings,
            index=[f"{ntype}-{idx}" for idx in mapping.keys()],
            columns=[f"dim_{i}" for i in range(embeddings.shape[1])],
        )

        embedding_df.to_csv(os.path.join(model_dir, f"{ntype}_embeddings.csv"))


if __name__ == "__main__":
    main()
