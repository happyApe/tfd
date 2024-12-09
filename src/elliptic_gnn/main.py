import argparse
import os
from datetime import datetime

import torch
from datasets import EllipticDataset
from models import GAT, GCN, GIN
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or visualize fraud detection models"
    )

    # Dataset arguments
    parser.add_argument(
        "--features-path",
        type=str,
        default="data/elliptic_bitcoin_dataset/elliptic_txs_features.csv",
        help="Path to features CSV file",
    )
    parser.add_argument(
        "--edges-path",
        type=str,
        default="data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv",
        help="Path to edges CSV file",
    )
    parser.add_argument(
        "--classes-path",
        type=str,
        default="data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv",
        help="Path to classes CSV file",
    )

    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="gat",
        choices=["gat", "gcn", "gin"],
        help="Type of GNN model to use",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=128, help="Hidden dimension size"
    )
    parser.add_argument(
        "--num-heads", type=int, default=2, help="Number of attention heads (GAT only)"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--print-freq",
        type=int,
        default=10,
        help="How often to print training progress",
    )

    # Visualization arguments
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the network instead of training",
    )
    parser.add_argument(
        "--time-step", type=int, default=30, help="Time step to visualize"
    )
    parser.add_argument(
        "--weights-path", type=str, help="Path to model weights for visualization"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory for saving results"
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default="runs",
        help="Directory for tensorboard logs",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dataset
    dataset = EllipticDataset(
        features_path=args.features_path,
        edges_path=args.edges_path,
        classes_path=args.classes_path,
    )

    # Get input dimension from dataset
    input_dim = dataset.pyg_dataset().num_node_features

    # Initialize model
    model_classes = {"gat": GAT, "gcn": GCN, "gin": GIN}
    model_class = model_classes[args.model_type]

    if args.model_type == "gat":
        model = model_class(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
    else:
        model = model_class(input_dim=input_dim, hidden_dim=args.hidden_dim)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        tensorboard_dir=os.path.join(
            args.tensorboard_dir, f"{args.model_type}_{timestamp}"
        ),
    )

    if args.visualize:
        if args.weights_path:
            trainer.load(args.weights_path)
        trainer.visualize(
            dataset=dataset,
            time_step=args.time_step,
            save_path=os.path.join(
                output_dir, f"visualization_step_{args.time_step}.png"
            ),
        )
    else:
        # Train the model
        print(f"Training {args.model_type.upper()} model...")
        best_f1 = trainer.train(num_epochs=args.epochs, print_freq=args.print_freq)
        print(f"\nTraining completed. Best validation F1: {best_f1:.4f}")

        # Save the model
        model_path = os.path.join(output_dir, f"{args.model_type}_model.pt")
        trainer.save(model_path)

        # Create visualization for the last epoch
        trainer.visualize(
            dataset=dataset,
            time_step=args.time_step,
            save_path=os.path.join(
                output_dir, f"final_visualization_step_{args.time_step}.png"
            ),
        )


if __name__ == "__main__":
    main()
