import json
import os
import sys

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


def load_model_and_data():
    global elliptic_dataset, ieee_cis_dataset, models, graphs

    # Load IEEE-CIS data and model
    try:
        cis_data_dir = "src/cis_gnn/data/ieee_cis_clean/"
        edge_files = [f for f in os.listdir(cis_data_dir) if f.startswith("relation")]

        g, features, target_id_to_node, id_to_node = construct_graph(
            cis_data_dir, edge_files, "features.csv", "TransactionID"
        )

        # Store the graph and mappings
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
        graphs["elliptic"] = data

    except Exception as e:
        print(f"Error loading Elliptic data: {e}")
        import traceback

        traceback.print_exc()

    return models, graphs


def convert_graph_to_json(g, dataset_type, time_step=30, sample_size=1000):
    """Convert graph to JSON format for D3"""
    if dataset_type == "ieee_cis":
        # Process IEEE-CIS heterogeneous graph
        nodes = []
        edges = []
        node_map = {}
        node_idx = 0

        # First add transaction nodes (they have features)
        transaction_type = "target"
        tx_nodes = g.nodes[transaction_type].data["features"]
        n_transactions = min(sample_size // 2, tx_nodes.shape[0])
        sampled_tx = np.random.choice(tx_nodes.shape[0], n_transactions, replace=False)

        # Add transaction nodes
        for idx in sampled_tx:
            node_map[f"{transaction_type}_{idx}"] = node_idx
            nodes.append(
                {
                    "id": node_idx,
                    "type": "Transaction",
                    "group": 0,
                    "color": "#1f77b4",  # blue for transactions
                }
            )
            node_idx += 1

        # Add connected entity nodes
        entity_colors = {
            "card": "#2ca02c",  # green
            "addr": "#d62728",  # red
            "email": "#9467bd",  # purple
            "id": "#8c564b",  # brown
            "device": "#e377c2",  # pink
        }

        for etype in g.canonical_etypes:
            src, rel, dst = etype
            if src == transaction_type or dst == transaction_type:
                u, v = g.edges(etype=rel)
                for i in range(len(u)):
                    src_node = f"{src}_{u[i].item()}"
                    dst_node = f"{dst}_{v[i].item()}"

                    # Add source node if not exists
                    if src_node not in node_map:
                        node_map[src_node] = node_idx
                        color = next(
                            (c for k, c in entity_colors.items() if k in src), "#aec7e8"
                        )
                        nodes.append(
                            {
                                "id": node_idx,
                                "type": src,
                                "group": len(nodes) % 5 + 1,
                                "color": color,
                            }
                        )
                        node_idx += 1

                    # Add target node if not exists
                    if dst_node not in node_map:
                        node_map[dst_node] = node_idx
                        color = next(
                            (c for k, c in entity_colors.items() if k in dst), "#aec7e8"
                        )
                        nodes.append(
                            {
                                "id": node_idx,
                                "type": dst,
                                "group": len(nodes) % 5 + 1,
                                "color": color,
                            }
                        )
                        node_idx += 1

                    # Add edge
                    edges.append(
                        {
                            "source": node_map[src_node],
                            "target": node_map[dst_node],
                            "type": rel,
                        }
                    )

                    if len(edges) >= sample_size:
                        break

    else:  # Elliptic dataset
        if not hasattr(g, "merged_df"):
            return {"nodes": [], "edges": []}

        # Get nodes for the specific time step
        node_list = g.merged_df.index[g.merged_df.loc[:, 1] == time_step].tolist()

        # Create nodes
        nodes = []
        node_map = {}

        for idx, node_id in enumerate(node_list):
            node_map[node_id] = idx

            if node_id in g.illicit_ids:
                color = "#d62728"  # red for fraud
                group = 1
            elif node_id in g.licit_ids:
                color = "#2ca02c"  # green for legitimate
                group = 0
            else:
                color = "#1f77b4"  # blue for unknown
                group = 2

            nodes.append(
                {
                    "id": idx,
                    "original_id": int(node_id),
                    "group": group,
                    "color": color,
                    "type": "Transaction",
                }
            )

        # Create edges
        edges = []
        edge_index = g.edge_index.cpu().numpy()

        for row in edge_index.T:
            source, target = row[0], row[1]
            if source in node_map and target in node_map:
                edges.append({"source": node_map[source], "target": node_map[target]})

            if len(edges) >= sample_size:
                break

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "nodeCount": len(nodes),
            "edgeCount": len(edges),
            "nodeTypes": list(set(n["type"] for n in nodes)),
        },
    }


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
