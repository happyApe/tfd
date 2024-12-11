# app.py
import glob
import json
import os

# Add parent directory to path to import from src
import sys
from datetime import datetime

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


def get_latest_model_path(base_path):
    """Get the latest model path based on timestamp in directory name"""
    dirs = glob.glob(os.path.join(base_path, "*"))
    if not dirs:
        return None
    latest_dir = max(dirs, key=os.path.getctime)
    return latest_dir


def load_model_and_data():
    models = {}
    graphs = {}

    # Load IEEE-CIS data and model
    try:
        # Correct paths for IEEE-CIS
        cis_data_dir = "src/cis_gnn/data/ieee_cis_clean/"  # Update this path
        edge_files = glob.glob(os.path.join(cis_data_dir, "relation*"))
        if edge_files:  # Only proceed if we found edge files
            edge_files = [os.path.basename(f) for f in edge_files]

            g, features, target_id_to_node, id_to_node = construct_graph(
                cis_data_dir, edge_files, "features.csv", "TransactionID"
            )

            # Find latest model directory
            cis_model_dir = get_latest_model_path("src/cis_gnn/model/")
            if cis_model_dir and os.path.exists(
                os.path.join(cis_model_dir, "model.pth")
            ):
                model_path = os.path.join(cis_model_dir, "model.pth")
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
                models["ieee_cis"] = model
                graphs["ieee_cis"] = g
    except Exception as e:
        print(f"Error loading IEEE-CIS data: {e}")

    try:
        # Load dataset
        elliptic_data_dir = "src/elliptic_gnn/data/elliptic_bitcoin_dataset/"
        elliptic_dataset = EllipticDataset(
            features_path=os.path.join(elliptic_data_dir, "elliptic_txs_features.csv"),
            edges_path=os.path.join(elliptic_data_dir, "elliptic_txs_edgelist.csv"),
            classes_path=os.path.join(elliptic_data_dir, "elliptic_txs_classes.csv"),
        )
        data = elliptic_dataset.pyg_dataset()

        # Load models from results directory
        model_paths = {
            "gat": "src/elliptic_gnn/results/gat_20241210_025231",
            "gcn": "src/elliptic_gnn/results/gcn_20241210_025601",
            "gin": "src/elliptic_gnn/results/gin_20241210_025836",
        }

        model_classes = {"gat": GAT, "gcn": GCN, "gin": GIN}

        for model_name, result_dir in model_paths.items():
            model_path = os.path.join(result_dir, f"{model_name}_model.pt")
            if os.path.exists(model_path):
                # Load the full checkpoint
                checkpoint = torch.load(model_path)

                # Initialize the model
                model = model_classes[model_name](input_dim=data.num_features)

                # Load just the model state dict
                model.load_state_dict(checkpoint["model_state_dict"])
                models[f"elliptic_{model_name}"] = model

        graphs["elliptic"] = data
    except Exception as e:
        print(f"Error loading Elliptic data: {e}")
        import traceback

        traceback.print_exc()  # This will print the full error traceback

    return models, graphs


def convert_graph_to_json(g, dataset_type, sample_size=1000):
    """Convert graph to JSON format for D3"""
    if dataset_type == "ieee_cis":
        # IEEE-CIS: Focus on transactions and their connections
        nodes = []
        edges = []
        node_map = {}
        node_idx = 0

        # Get all transaction nodes first
        transaction_type = (
            "target"  # This is how transactions are labeled in the HeteroRGCN
        )
        n_transactions = g.number_of_nodes(transaction_type)
        sampled_transactions = np.random.choice(
            n_transactions, size=min(sample_size // 2, n_transactions), replace=False
        )

        # Add transaction nodes
        for idx in sampled_transactions:
            node_map[f"{transaction_type}_{idx}"] = node_idx
            nodes.append({"id": node_idx, "type": "Transaction", "group": 0})
            node_idx += 1

        # Add connected nodes from other types
        for etype in g.canonical_etypes:
            src, rel, dst = etype
            if src == transaction_type or dst == transaction_type:
                u, v = g.edges(etype=rel)
                u = u.cpu().numpy()
                v = v.cpu().numpy()

                for i in range(len(u)):
                    if f"{src}_{u[i]}" in node_map or f"{dst}_{v[i]}" in node_map:
                        # Add source node if not exists
                        if f"{src}_{u[i]}" not in node_map:
                            node_map[f"{src}_{u[i]}"] = node_idx
                            nodes.append(
                                {
                                    "id": node_idx,
                                    "type": src,
                                    "group": g.ntypes.index(src),
                                }
                            )
                            node_idx += 1

                        # Add target node if not exists
                        if f"{dst}_{v[i]}" not in node_map:
                            node_map[f"{dst}_{v[i]}"] = node_idx
                            nodes.append(
                                {
                                    "id": node_idx,
                                    "type": dst,
                                    "group": g.ntypes.index(dst),
                                }
                            )
                            node_idx += 1

                        # Add edge
                        edges.append(
                            {
                                "source": node_map[f"{src}_{u[i]}"],
                                "target": node_map[f"{dst}_{v[i]}"],
                                "type": rel,
                            }
                        )

    else:  # Elliptic dataset
        nodes = []
        edges = []
        node_map = {}

        # Get edge index and convert to numpy for easier handling
        edge_index = g.edge_index.cpu().numpy()
        labels = g.y.cpu().numpy()

        # Start with a random node and do BFS to get connected components
        all_nodes = set(range(g.num_nodes))
        sampled_nodes = set()

        while len(sampled_nodes) < sample_size and all_nodes:
            # Pick a random start node
            start_node = np.random.choice(list(all_nodes - sampled_nodes))

            # Do BFS
            queue = [start_node]
            component = set()

            while queue and len(component) < sample_size - len(sampled_nodes):
                current = queue.pop(0)
                if current not in component:
                    component.add(current)
                    # Find neighbors
                    mask = (edge_index[0] == current) | (edge_index[1] == current)
                    neighbors = set(edge_index[0][mask]) | set(edge_index[1][mask])
                    queue.extend(neighbors - component)

            sampled_nodes.update(component)
            if len(sampled_nodes) >= sample_size:
                break

        # Convert sampled nodes to list and create mapping
        sampled_nodes = list(sampled_nodes)
        node_map = {node: idx for idx, node in enumerate(sampled_nodes)}

        # Create nodes
        for node in sampled_nodes:
            nodes.append(
                {
                    "id": node_map[node],
                    "group": int(labels[node]),
                    "original_id": int(node),
                }
            )

        # Create edges
        for i in range(edge_index.shape[1]):
            source, target = edge_index[0, i], edge_index[1, i]
            if source in node_map and target in node_map:
                edges.append({"source": node_map[source], "target": node_map[target]})

    return {"nodes": nodes, "edges": edges}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/graph/<dataset>")
def get_graph(dataset):
    try:
        if dataset in graphs:
            graph_data = convert_graph_to_json(graphs[dataset], dataset)
            return jsonify(graph_data)
        return jsonify({"error": "Dataset not found"}), 404
    except Exception as e:
        print(f"Error converting graph to JSON: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Load models and graphs globally
    print("Loading models and graphs...")
    models, graphs = load_model_and_data()
    print("Loaded models:", list(models.keys()))
    print("Loaded graphs:", list(graphs.keys()))
    app.run(debug=False)
