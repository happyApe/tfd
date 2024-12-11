# app.py
import json
import os

import dgl
import networkx as nx
import torch
from flask import Flask, jsonify, render_template

from cis_gnn.graph_utils import construct_graph
from cis_gnn.model import HeteroRGCN
from elliptic_gnn.datasets import EllipticDataset
from elliptic_gnn.models import GAT, GCN, GIN

app = Flask(__name__)


def load_model_and_data():
    models = {}
    graphs = {}

    # Load IEEE-CIS data and model
    try:
        cis_data_dir = "data/ieee_cis_clean/"
        g, features, target_id_to_node, id_to_node = construct_graph(
            cis_data_dir, ["relation*"], "features.csv", "TransactionID"
        )

        # Load model
        model_path = "model/ieee_cis/model.pth"
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
            models["ieee_cis"] = model
            graphs["ieee_cis"] = g
    except Exception as e:
        print(f"Error loading IEEE-CIS data: {e}")

    # Load Elliptic data and models
    try:
        elliptic_dataset = EllipticDataset()
        data = elliptic_dataset.pyg_dataset()

        model_types = {"gat": GAT, "gcn": GCN, "gin": GIN}

        for model_name, model_class in model_types.items():
            model_path = f"results/{model_name}_model.pt"
            if os.path.exists(model_path):
                model = model_class(input_dim=data.num_features)
                model.load_state_dict(torch.load(model_path))
                models[f"elliptic_{model_name}"] = model

        graphs["elliptic"] = data
    except Exception as e:
        print(f"Error loading Elliptic data: {e}")

    return models, graphs


models, graphs = load_model_and_data()


def convert_graph_to_json(g, dataset_type):
    """Convert graph to JSON format for D3"""
    if dataset_type == "ieee_cis":
        # Convert heterogeneous graph
        nx_g = nx.Graph()
        nodes = []
        edges = []

        # Add nodes
        node_idx = 0
        node_map = {}

        for ntype in g.ntypes:
            for i in range(g.number_of_nodes(ntype)):
                node_map[f"{ntype}_{i}"] = node_idx
                nodes.append(
                    {"id": node_idx, "type": ntype, "group": g.ntypes.index(ntype)}
                )
                node_idx += 1

        # Add edges
        for etype in g.canonical_etypes:
            src, rel, dst = etype
            u, v = g.edges(etype=rel)
            for i in range(len(u)):
                edges.append(
                    {
                        "source": node_map[f"{src}_{u[i].item()}"],
                        "target": node_map[f"{dst}_{v[i].item()}"],
                        "type": rel,
                    }
                )

    else:  # Elliptic dataset
        nodes = [{"id": i, "group": int(g.y[i].item())} for i in range(g.num_nodes)]
        edges = [
            {"source": u.item(), "target": v.item()} for u, v in zip(*g.edge_index)
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
    app.run(debug=True)
