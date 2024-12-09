import copy
import os
import pickle
import sys
import time

import dgl
import numpy as np
import torch
import torch.optim
from graph_utils import construct_graph, get_edgelists, get_labels
from model import HeteroRGCN
from sklearn.metrics import confusion_matrix
from utils import get_logger, get_metrics, parse_args

# Set up environment
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)
os.environ["DGLBACKEND"] = "pytorch"


def initial_record():
    """Initialize results file."""
    if os.path.exists("./output/results.txt"):
        os.remove("./output/results.txt")
    with open("./output/results.txt", "w") as f:
        f.write("Epoch,Time(s),Loss,F1\n")


def normalize(feature_matrix):
    """Normalize features using mean and standard deviation."""
    mean = torch.mean(feature_matrix, axis=0)
    stdev = torch.sqrt(
        torch.sum((feature_matrix - mean) ** 2, axis=0) / feature_matrix.shape[0]
    )
    return mean, stdev, (feature_matrix - mean) / stdev


def train_fg(
    model,
    optim,
    loss_fn,
    features,
    labels,
    train_g,
    test_g,
    test_mask,
    device,
    n_epochs,
    thresh,
    compute_metrics=True,
):
    """Train model using full graph batching."""
    duration = []
    best_loss = float("inf")
    best_model = None

    for epoch in range(n_epochs):
        tic = time.time()

        # Forward pass
        pred = model(train_g, features.to(device))
        loss_val = loss_fn(pred, labels)

        # Backward pass
        optim.zero_grad()
        loss_val.backward()
        optim.step()

        duration.append(time.time() - tic)
        metric = evaluate(model, train_g, features, labels, device)

        print(
            f"Epoch {epoch:05d}, Time(s) {np.mean(duration):.4f}, "
            f"Loss {loss_val:.4f}, F1 {metric:.4f}"
        )

        # Save results
        with open("./output/results.txt", "a+") as f:
            f.write(
                f"{epoch:05d},{np.mean(duration):.4f},{loss_val:.4f},{metric:.4f}\n"
            )

        # Save best model
        if loss_val < best_loss:
            best_loss = loss_val
            best_model = copy.deepcopy(model)

    # Get predictions
    class_preds, pred_proba = get_model_class_predictions(
        best_model, test_g, features, labels, device, threshold=thresh
    )

    # Compute metrics
    if compute_metrics:
        metrics = get_metrics(
            class_preds, pred_proba, labels.numpy(), test_mask.numpy(), "./output/"
        )
        print_metrics(*metrics)

    return best_model, class_preds, pred_proba


def evaluate(model, g, features, labels, device):
    """Compute F1 score for binary classification."""
    preds = model(g, features.to(device))
    preds = torch.argmax(preds, axis=1).cpu().numpy()
    return compute_f1_score(labels.cpu().numpy(), preds)


def compute_f1_score(y_true, y_pred):
    """Compute F1 score from confusion matrix."""
    cf_m = confusion_matrix(y_true, y_pred)
    precision = cf_m[1, 1] / (cf_m[1, 1] + cf_m[0, 1] + 1e-5)
    recall = cf_m[1, 1] / (cf_m[1, 1] + cf_m[1, 0])
    f1 = 2 * (precision * recall) / (precision + recall + 1e-5)
    return f1


def get_model_class_predictions(model, g, features, labels, device, threshold=None):
    """Get model predictions and probabilities."""
    with torch.no_grad():
        logits = model(g, features.to(device))
        probs = torch.softmax(logits, dim=-1)

        if threshold is None:
            preds = logits.argmax(axis=1).cpu().numpy()
        else:
            preds = (probs[:, 1] > threshold).cpu().numpy()

        return preds, probs[:, 1].cpu().numpy()


def save_model(g, model, model_dir, id_to_node, mean, stdev):
    """Save model and metadata."""
    # Save model parameters
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

    # Save metadata
    metadata = {
        "etypes": g.canonical_etypes,
        "ntype_cnt": {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes},
        "feat_mean": mean,
        "feat_std": stdev,
    }
    with open(os.path.join(model_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    # Save node embeddings
    for ntype, mapping in id_to_node.items():
        if ntype == "target":
            continue

        # Get node mappings
        old_ids = list(mapping.keys())
        node_ids = list(mapping.values())

        # Get embeddings
        embeddings = model.embed[ntype].detach().numpy()

        # Create DataFrames
        node_df = pd.DataFrame(
            {
                "~label": [ntype] * len(old_ids),
                "~id_tmp": old_ids,
                "~id": [f"{ntype}-{id_}" for id_ in old_ids],
                "node_id": node_ids,
            }
        )

        # Create feature columns
        feat_df = pd.DataFrame(
            {f"val{i+1}:Double": embeddings[:, i] for i in range(embeddings.shape[1])}
        )

        # Merge and save
        result_df = node_df.merge(feat_df, left_on="node_id", right_index=True)
        result_df = result_df.drop(["~id_tmp", "node_id"], axis=1)
        result_df.to_csv(
            os.path.join(model_dir, f"{ntype}.csv"),
            index=False,
            header=True,
            encoding="utf-8",
        )


def print_metrics(acc, f1, precision, recall, roc_auc, pr_auc, ap, cm):
    """Print model evaluation metrics."""
    print("\nMetrics:")
    print(f"Confusion Matrix:\n{cm}")
    print(f"F1: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Average Precision: {ap:.4f}")


def main():
    args = parse_args()
    print(f"Running with args: {args}")

    # Set up device
    device = torch.device("cuda:0" if args.num_gpus else "cpu")

    # Get edge lists and construct graph
    args.edges = get_edgelists("relation*", args.training_dir)
    g, features, target_id_to_node, id_to_node = construct_graph(
        args.training_dir, args.edges, args.nodes, args.target_ntype
    )

    # Normalize features
    mean, stdev, features = normalize(torch.from_numpy(features))
    g.nodes["target"].data["features"] = features

    # Get labels and masks
    n_nodes = g.number_of_nodes("target")
    labels, _, test_mask = get_labels(
        target_id_to_node,
        n_nodes,
        args.target_ntype,
        os.path.join(args.training_dir, args.labels),
        os.path.join(args.training_dir, args.new_accounts),
    )

    # Convert to tensors
    labels = torch.from_numpy(labels).long().to(device)
    test_mask = torch.from_numpy(test_mask).to(device)
    features = features.to(device)

    # Print data statistics
    print_data_stats(g, features, test_mask)

    # Initialize model
    in_feats = features.shape[1]
    n_classes = 2
    ntype_dict = {n_type: g.number_of_nodes(n_type) for n_type in g.ntypes}

    model = HeteroRGCN(
        ntype_dict,
        g.etypes,
        in_feats,
        args.n_hidden,
        n_classes,
        args.n_layers,
        in_feats,
    ).to(device)

    # Set up training
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train model
    print("Starting training...")
    initial_record()
    model, class_preds, pred_proba = train_fg(
        model,
        optimizer,
        loss_fn,
        features,
        labels,
        g,
        g,
        test_mask,
        device,
        args.n_epochs,
        args.threshold,
        args.compute_metrics,
    )

    # Save model
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    save_model(g, model, args.model_dir, id_to_node, mean, stdev)
    print("Training completed. Model and metadata saved.")


def print_data_stats(g, features, test_mask):
    """Print dataset statistics."""
    n_nodes = sum(g.number_of_nodes(ntype) for ntype in g.ntypes)
    n_edges = sum(g.number_of_edges(etype) for etype in g.etypes)

    print("\nDataset Statistics:")
    print(f"Number of Nodes: {n_nodes}")
    print(f"Number of Edges: {n_edges}")
    print(f"Feature Shape: {features.shape}")
    print(f"Number of Test Samples: {test_mask.sum()}\n")


if __name__ == "__main__":
    main()
