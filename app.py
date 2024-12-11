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
    """Convert graph to JSON format for D3"""

    def tensor_to_list(tensor):
        """Convert a tensor to a Python list"""
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().detach().numpy().tolist()
        return tensor

    if dataset_type == "ieee_cis":
        # Convert heterogeneous graph with sampling
        nodes = []
        edges = []
        node_map = {}
        node_idx = 0

        # Sample nodes from each type
        for ntype in g.ntypes:
            n_type_nodes = g.number_of_nodes(ntype)
            sample_size_type = min(sample_size // len(g.ntypes), n_type_nodes)
            sampled_indices = torch.randperm(n_type_nodes)[:sample_size_type]

            for i in sampled_indices:
                node_map[f"{ntype}_{i.item()}"] = node_idx
                nodes.append(
                    {
                        "id": int(node_idx),  # Convert to int
                        "type": str(ntype),
                        "group": g.ntypes.index(ntype),
                    }
                )
                node_idx += 1

        # Sample edges
        for etype in g.canonical_etypes:
            src, rel, dst = etype
            u, v = g.edges(etype=rel)
            edge_count = 0

            for i in range(min(len(u), sample_size // len(g.canonical_etypes))):
                src_key = f"{src}_{u[i].item()}"
                dst_key = f"{dst}_{v[i].item()}"

                if src_key in node_map and dst_key in node_map:
                    edges.append(
                        {
                            "source": int(node_map[src_key]),  # Convert to int
                            "target": int(node_map[dst_key]),
                            "type": str(rel),
                        }
                    )
                    edge_count += 1

                if edge_count >= sample_size // len(g.canonical_etypes):
                    break

    else:  # Elliptic dataset
        # Sample nodes
        n_nodes = g.num_nodes
        sampled_nodes = torch.randperm(n_nodes)[:sample_size]
        nodes = []
        node_map = {}

        # Create nodes with integer indices
        for i, node_idx in enumerate(sampled_nodes):
            node_map[node_idx.item()] = i
            nodes.append({"id": i, "group": int(tensor_to_list(g.y[node_idx]))})

        # Create edges between sampled nodes
        edges = []
        edge_index = g.edge_index

        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()

            if src in node_map and dst in node_map:
                edges.append({"source": node_map[src], "target": node_map[dst]})

            if len(edges) >= sample_size * 2:  # Limit number of edges
                break

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
