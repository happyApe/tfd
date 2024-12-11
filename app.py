# app.py
import glob
import json
import os

# Add parent directory to path to import from src
import sys
from datetime import datetime

import dgl
import networkx as nx
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
    """Convert graph to JSON format for D3 with sampling for large graphs"""
    if dataset_type == "ieee_cis":
        # Convert heterogeneous graph with sampling
        nx_g = nx.Graph()
        nodes = []
        edges = []

        # Sample nodes from each type
        node_idx = 0
        node_map = {}

        for ntype in g.ntypes:
            n_type_nodes = g.number_of_nodes(ntype)
            sample_size_type = min(sample_size // len(g.ntypes), n_type_nodes)
            sampled_indices = torch.randperm(n_type_nodes)[:sample_size_type]

            for i in sampled_indices:
                node_map[f"{ntype}_{i.item()}"] = node_idx
                nodes.append(
                    {"id": node_idx, "type": ntype, "group": g.ntypes.index(ntype)}
                )
                node_idx += 1

        # Sample edges between sampled nodes
        for etype in g.canonical_etypes:
            src, rel, dst = etype
            u, v = g.edges(etype=rel)
            edge_mask = torch.zeros(len(u), dtype=torch.bool)

            for i, (src_idx, dst_idx) in enumerate(zip(u, v)):
                src_key = f"{src}_{src_idx.item()}"
                dst_key = f"{dst}_{dst_idx.item()}"
                if src_key in node_map and dst_key in node_map:
                    edge_mask[i] = True

            sampled_edges = torch.where(edge_mask)[0][
                : sample_size // len(g.canonical_etypes)
            ]

            for i in sampled_edges:
                src_key = f"{src}_{u[i].item()}"
                dst_key = f"{dst}_{v[i].item()}"
                edges.append(
                    {
                        "source": node_map[src_key],
                        "target": node_map[dst_key],
                        "type": rel,
                    }
                )

    else:  # Elliptic dataset
        # Sample nodes
        n_nodes = g.num_nodes
        sampled_nodes = torch.randperm(n_nodes)[:sample_size]
        nodes = [{"id": i, "group": int(g.y[i].item())} for i in sampled_nodes]

        # Sample edges between sampled nodes
        edge_index = g.edge_index
        edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        node_set = set(sampled_nodes.tolist())

        for i in range(edge_index.size(1)):
            if (
                edge_index[0, i].item() in node_set
                and edge_index[1, i].item() in node_set
            ):
                edge_mask[i] = True

        sampled_edges = torch.where(edge_mask)[0][: sample_size * 2]
        edges = [
            {"source": edge_index[0, i].item(), "target": edge_index[1, i].item()}
            for i in sampled_edges
        ]

    return {"nodes": nodes, "edges": edges}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/graph/<dataset>")
def get_graph(dataset):
    if dataset in graphs:
        return jsonify(convert_graph_to_json(graphs[dataset], dataset))
    return jsonify({"error": "Dataset not found"}), 404


if __name__ == "__main__":
    # Load models and graphs globally
    print("Loading models and graphs...")
    models, graphs = load_model_and_data()
    print("Loaded models:", list(models.keys()))
    print("Loaded graphs:", list(graphs.keys()))
    app.run(debug=True)
