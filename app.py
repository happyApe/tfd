import glob
import json
import os
import sys
import traceback

import dgl
import networkx as nx
import numpy as np
import torch
from flask import Flask, jsonify, render_template

sys.path.append("src")

from cis_gnn.graph_utils import construct_graph
from cis_gnn.model import HeteroRGCN
from elliptic_gnn.datasets import EllipticDataset
from elliptic_gnn.models import GAT, GCN, GIN

app = Flask(__name__)

# Global variables to store datasets and models
elliptic_dataset = None
ieee_cis_dataset = None
models = {}
graphs = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_data():
    global elliptic_dataset, ieee_cis_dataset, models, graphs, device

    # Load IEEE-CIS data and model
    try:
        cis_data_dir = "src/cis_gnn/data/ieee_cis_clean/"
        edge_files = [f for f in os.listdir(cis_data_dir) if f.startswith("relation")]

        g, features, target_id_to_node, id_to_node = construct_graph(
            cis_data_dir, edge_files, "features.csv", "TransactionID"
        )

        # Load latest IEEE-CIS model
        cis_model_dir = sorted(glob.glob("src/cis_gnn/model/*"))[-1]
        model_path = os.path.join(cis_model_dir, "model.pth")
        if os.path.exists(model_path):
            model = HeteroRGCN(
                {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes},
                g.etypes,
                features.shape[1],
                16,
                2,
                3,
                features.shape[1],
            )
            model.load_state_dict(torch.load(model_path))
            models["ieee_cis"] = model.eval()

        ieee_cis_dataset = {
            "graph": g,
            "features": features,
            "target_id_to_node": target_id_to_node,
            "id_to_node": id_to_node,
        }
        graphs["ieee_cis"] = g

    except Exception as e:
        print(f"Error loading IEEE-CIS data: {e}")
        import traceback

        traceback.print_exc()

    # Load Elliptic data and models
    try:
        elliptic_data_dir = "src/elliptic_gnn/data/elliptic_bitcoin_dataset/"
        elliptic_dataset = EllipticDataset(
            features_path=os.path.join(elliptic_data_dir, "elliptic_txs_features.csv"),
            edges_path=os.path.join(elliptic_data_dir, "elliptic_txs_edgelist.csv"),
            classes_path=os.path.join(elliptic_data_dir, "elliptic_txs_classes.csv"),
        )
        data = elliptic_dataset.pyg_dataset()

        # Load Elliptic models
        model_paths = {
            "gat": "src/elliptic_gnn/results/gat_20241210_025231/gat_model.pt",
            "gcn": "src/elliptic_gnn/results/gcn_20241210_025601/gcn_model.pt",
            "gin": "src/elliptic_gnn/results/gin_20241210_025836/gin_model.pt",
        }

        model_classes = {"gat": GAT, "gcn": GCN, "gin": GIN}

        for model_name, model_path in model_paths.items():
            if os.path.exists(model_path):
                model = model_classes[model_name](input_dim=data.num_features)
                model = model.float()
                model = model.to(device)
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                models[f"elliptic_{model_name}"] = model

        graphs["elliptic"] = data

    except Exception as e:
        print(f"Error loading Elliptic data: {e}")
        import traceback

        traceback.print_exc()

    return models, graphs


def get_predictions(dataset_type, graph_data):
    """Get model predictions for the nodes"""
    if dataset_type == "ieee_cis":
        if "ieee_cis" in models:
            model = models["ieee_cis"]
            with torch.no_grad():
                pred = model(graphs["ieee_cis"], ieee_cis_dataset["features"])
                pred_probs = torch.softmax(pred, dim=1)
                predictions = torch.argmax(pred_probs, dim=1).cpu().numpy()
                return predictions
    else:  # Elliptic dataset
        if "elliptic_gat" in models:  # Using GAT model for predictions
            model = models["elliptic_gat"]
            data = graphs["elliptic"].to(model.device)
            with torch.no_grad():
                pred_scores, pred_labels = model.test(data, labeled_only=True)
                return pred_labels.cpu().numpy()
    return None


def convert_graph_to_json(g, dataset_type, time_step=30, sample_size=1000):
    """Convert graph to JSON format for D3 with both true labels and predictions"""
    if dataset_type == "ieee_cis":
        # Get predictions for IEEE-CIS dataset
        predictions = None
        if "ieee_cis" in models:
            model = models["ieee_cis"]
            with torch.no_grad():
                pred = model(g, ieee_cis_dataset["features"])
                pred_probs = torch.softmax(pred, dim=1)
                predictions = torch.argmax(pred_probs, dim=1).cpu().numpy()

        nodes = []
        edges = []
        node_map = {}
        node_idx = 0

        # First add transaction nodes
        transaction_type = "target"
        tx_nodes = g.nodes[transaction_type].data["features"]
        n_transactions = min(sample_size // 2, tx_nodes.shape[0])
        sampled_tx = np.random.choice(tx_nodes.shape[0], n_transactions, replace=False)

        # Add transaction nodes with predictions
        for idx in sampled_tx:
            pred = predictions[idx] if predictions is not None else None
            node_map[f"{transaction_type}_{idx}"] = node_idx
            nodes.append(
                {
                    "id": node_idx,
                    "type": "Transaction",
                    "group": 0,
                    "true_color": "#1f77b4",  # Default blue for transactions
                    "pred_color": (
                        "#d62728"
                        if pred == 1
                        else "#2ca02c" if pred == 0 else "#1f77b4"
                    ),
                    "predicted": bool(pred) if pred is not None else None,
                }
            )
            node_idx += 1

        # Entity type colors
        entity_colors = {
            "card": "#ff7f0e",  # Orange
            "addr": "#9467bd",  # Purple
            "email": "#8c564b",  # Brown
            "id": "#e377c2",  # Pink
            "device": "#7f7f7f",  # Gray
            "ProductCD": "#bcbd22",  # Yellow-green
            "target": "#17becf",  # Cyan
        }

        # Add connected entity nodes
        for etype in g.canonical_etypes:
            src, rel, dst = etype
            if src == transaction_type or dst == transaction_type:
                u, v = g.edges(etype=rel)
                u = u.cpu().numpy()
                v = v.cpu().numpy()

                for i in range(len(u)):
                    src_key = f"{src}_{u[i]}"
                    dst_key = f"{dst}_{v[i]}"

                    # Add source node if not exists
                    if src_key not in node_map:
                        node_map[src_key] = node_idx
                        color = entity_colors.get(src, "#aec7e8")
                        nodes.append(
                            {
                                "id": node_idx,
                                "type": src,
                                "group": g.ntypes.index(src),
                                "true_color": color,
                                "pred_color": color,
                            }
                        )
                        node_idx += 1

                    # Add target node if not exists
                    if dst_key not in node_map:
                        node_map[dst_key] = node_idx
                        color = entity_colors.get(dst, "#aec7e8")
                        nodes.append(
                            {
                                "id": node_idx,
                                "type": dst,
                                "group": g.ntypes.index(dst),
                                "true_color": color,
                                "pred_color": color,
                            }
                        )
                        node_idx += 1

                    # Add edge
                    edges.append(
                        {
                            "source": node_map[src_key],
                            "target": node_map[dst_key],
                            "type": rel,
                        }
                    )

                    if len(edges) >= sample_size:
                        break

    else:  # Elliptic dataset
        # Get predictions from GAT model
        predictions = None
        if "elliptic_gat" in models:
            model = models["elliptic_gat"]
            data = graphs["elliptic"].to(device)

            with torch.no_grad():
                try:
                    # Convert data to float32 to match model parameters
                    data.x = data.x.float()  # Convert from double to float
                    data.edge_index = data.edge_index.long()  # Ensure indices are long

                    # Move data to device
                    data.x = data.x.to(device)
                    data.edge_index = data.edge_index.to(device)

                    # Get predictions
                    pred_scores = model(data)
                    predictions = (pred_scores > 0.5).cpu().numpy().flatten()

                    print(f"Predictions shape: {predictions.shape}")
                    print(f"Number of fraud predictions: {np.sum(predictions == 1)}")

                except Exception as e:
                    print(f"Error getting predictions: {e}")
                    traceback.print_exc()
                    predictions = None
        # Get nodes for the specific time step
        node_list = g.merged_df.index[g.merged_df.loc[:, 1] == time_step].tolist()
        print(f"Total nodes for time step {time_step}: {len(node_list)}")

        # Create nodes with true labels and predictions
        nodes = []
        node_map = {}

        for idx, node_id in enumerate(node_list):
            node_map[node_id] = idx

            # Get prediction for this node if available
            pred = predictions[node_id] if predictions is not None else None

            # Set true color based on actual label
            if node_id in g.illicit_ids:
                true_color = "#d62728"  # Red for true fraud
                group = 1
            elif node_id in g.licit_ids:
                true_color = "#2ca02c"  # Green for legitimate
                group = 0
            else:
                true_color = "#1f77b4"  # Blue for unknown
                group = 2

            # Set predicted color based on model prediction
            if pred is not None:
                pred_color = (
                    "#d62728" if pred else "#2ca02c"
                )  # Red for fraud, green for legitimate
            else:
                pred_color = true_color

            nodes.append(
                {
                    "id": idx,
                    "original_id": int(node_id),
                    "group": group,
                    "true_color": true_color,
                    "pred_color": pred_color,
                    "type": "Transaction",
                    "predicted": bool(pred) if pred is not None else None,
                    "is_fraud": group == 1,
                    "predicted_fraud": bool(pred) if pred is not None else None,
                }
            )

        # Create edges
        edges = []
        edge_index = g.edge_index.cpu().numpy()

        for i in range(edge_index.shape[1]):
            source, target = edge_index[0, i], edge_index[1, i]
            if source in node_map and target in node_map:
                edges.append({"source": node_map[source], "target": node_map[target]})

            if len(edges) >= sample_size * 2:
                break

    stats = {
        "nodeCount": len(nodes),
        "edgeCount": len(edges),
        "nodeTypes": list(set(n["type"] for n in nodes)),
        "predictionStats": {
            "total": len(nodes),
            "predicted": sum(1 for n in nodes if n.get("predicted") is not None),
        },
    }

    return {"nodes": nodes, "edges": edges, "stats": stats}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/graph/<dataset>")
def get_graph(dataset):
    try:
        if dataset in graphs:
            sample_size = 200 if dataset == "ieee_cis" else 500
            time_step = 30  # Default time step for Elliptic dataset
            graph_data = convert_graph_to_json(
                elliptic_dataset if dataset == "elliptic" else graphs[dataset],
                dataset,
                time_step=time_step,
                sample_size=sample_size,
            )

            # Add debugging information
            if dataset == "elliptic":
                fraud_count = sum(
                    1 for node in graph_data["nodes"] if node.get("predicted_fraud")
                )
                true_fraud_count = sum(
                    1 for node in graph_data["nodes"] if node.get("is_fraud")
                )
                print(f"Number of predicted fraud cases: {fraud_count}")
                print(f"Number of true fraud cases: {true_fraud_count}")

            return jsonify(graph_data)
        return jsonify({"error": "Dataset not found"}), 404
    except Exception as e:
        print(f"Error converting graph to JSON: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Loading models and graphs...")
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        models, graphs = load_model_and_data()
        print("Loaded models:", list(models.keys()))
        print("Loaded graphs:", list(graphs.keys()))
    app.run(debug=False, host="0.0.0.0")
