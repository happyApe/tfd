import logging
import os
import re

import dgl
import numpy as np
import pandas as pd
import torch

from .utils import get_logger


def get_features(id_to_node, node_features, device):
    """
    Get node features from file and map to graph nodes.

    Args:
        id_to_node (dict): Dictionary mapping node names(id) to dgl node idx
        node_features (str): Path to file containing node features
        device (torch.device): Device to store tensors on

    Returns:
        tuple: (feature matrix, list of new nodes)
    """
    indices, features, new_nodes = [], [], []
    max_node = max(id_to_node.values())

    def convert_feature(x):
        """Convert feature value to float, handling special cases."""
        if x.lower() == "false":
            return 0.0
        elif x.lower() == "true":
            return 1.0
        try:
            return float(x)
        except ValueError:
            return 0.0

    with open(node_features, "r") as fh:
        for line in fh:
            node_feats = line.strip().split(",")
            node_id = node_feats[0]

            feats = np.array([convert_feature(x) for x in node_feats[1:]])
            features.append(feats)

            if node_id not in id_to_node:
                max_node += 1
                id_to_node[node_id] = max_node
                new_nodes.append(max_node)

            indices.append(id_to_node[node_id])

    features = np.array(features).astype("float32")
    features = features[np.argsort(indices), :]

    # Convert to torch tensor and move to device
    features = torch.from_numpy(features).to(device)

    print(f"Loaded features shape: {features.shape}")
    print(f"Number of unique nodes: {len(set(indices))}")

    return features, new_nodes


def get_labels(
    id_to_node,
    n_nodes,
    target_node_type,
    labels_path,
    masked_nodes_path,
    additional_mask_rate=0,
    device=None,
):
    """
    Get node labels and masks for training/testing.

    Args:
        id_to_node (dict): Dictionary mapping node names to dgl node idx
        n_nodes (int): Number of nodes in the graph
        target_node_type (str): Column name for target node type
        labels_path (str): Path to file containing labeled nodes
        masked_nodes_path (str): Path to file containing nodes to mask
        additional_mask_rate (float): Additional masking rate for training nodes
        device (torch.device): Device to store tensors on

    Returns:
        tuple: (labels, train_mask, test_mask)
    """
    node_to_id = {v: k for k, v in id_to_node.items()}
    user_to_label = pd.read_csv(labels_path).set_index(target_node_type)
    labels = user_to_label.loc[
        map(int, pd.Series(node_to_id)[np.arange(n_nodes)].values)
    ].values.flatten()

    masked_nodes = read_masked_nodes(masked_nodes_path)
    train_mask, test_mask = _get_mask(
        id_to_node, node_to_id, n_nodes, masked_nodes, additional_mask_rate
    )

    # Convert to torch tensors and move to device if specified
    labels = torch.from_numpy(labels)
    train_mask = torch.from_numpy(train_mask)
    test_mask = torch.from_numpy(test_mask)

    if device is not None:
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        test_mask = test_mask.to(device)

    return labels, train_mask, test_mask


def read_masked_nodes(masked_nodes_path):
    """Read list of masked nodes from file."""
    with open(masked_nodes_path, "r") as fh:
        masked_nodes = [line.strip() for line in fh]
    return masked_nodes


def _get_mask(id_to_node, node_to_id, num_nodes, masked_nodes, additional_mask_rate):
    """Generate train and test masks."""
    train_mask = np.ones(num_nodes)
    test_mask = np.zeros(num_nodes)

    for node_id in masked_nodes:
        train_mask[id_to_node[node_id]] = 0
        test_mask[id_to_node[node_id]] = 1

    if additional_mask_rate and additional_mask_rate < 1:
        unmasked = np.array(
            [idx for idx in range(num_nodes) if node_to_id[idx] not in masked_nodes]
        )
        yet_unmasked = np.random.permutation(unmasked)[
            : int(additional_mask_rate * num_nodes)
        ]
        train_mask[yet_unmasked] = 0

    return train_mask, test_mask


def _get_node_idx(id_to_node, node_type, node_id, ptr):
    """Get or create node index for a given node."""
    if node_type in id_to_node:
        if node_id in id_to_node[node_type]:
            node_idx = id_to_node[node_type][node_id]
        else:
            id_to_node[node_type][node_id] = ptr
            node_idx = ptr
            ptr += 1
    else:
        id_to_node[node_type] = {}
        id_to_node[node_type][node_id] = ptr
        node_idx = ptr
        ptr += 1
    return node_idx, id_to_node, ptr


def parse_edgelist(
    edges, id_to_node, header=False, source_type="user", sink_type="user"
):
    """Parse edge list file and create graph edges."""
    edge_list = []
    rev_edge_list = []
    source_pointer = sink_pointer = 0

    with open(edges, "r") as fh:
        for i, line in enumerate(fh):
            source, sink = line.strip().split(",")

            if i == 0:
                if header:
                    source_type, sink_type = source, sink
                if source_type in id_to_node:
                    source_pointer = max(id_to_node[source_type].values()) + 1
                if sink_type in id_to_node:
                    sink_pointer = max(id_to_node[sink_type].values()) + 1
                continue

            source_node, id_to_node, source_pointer = _get_node_idx(
                id_to_node, source_type, source, source_pointer
            )

            if source_type == sink_type:
                sink_node, id_to_node, source_pointer = _get_node_idx(
                    id_to_node, sink_type, sink, source_pointer
                )
            else:
                sink_node, id_to_node, sink_pointer = _get_node_idx(
                    id_to_node, sink_type, sink, sink_pointer
                )

            edge_list.append((source_node, sink_node))
            rev_edge_list.append((sink_node, source_node))

    return edge_list, rev_edge_list, id_to_node, source_type, sink_type


def get_edgelists(edgelist_expression, directory):
    """Get edge list files matching expression pattern."""
    if "," in edgelist_expression:
        return edgelist_expression.split(",")
    files = os.listdir(directory)
    compiled_expression = re.compile(edgelist_expression)
    return [filename for filename in files if compiled_expression.match(filename)]


def construct_graph(training_dir, edges, nodes, target_node_type, device=None):
    """
    Construct heterogeneous graph from edge lists and features.

    Args:
        training_dir (str): Directory containing training data
        edges (list): List of edge list files
        nodes (str): Node features file
        target_node_type (str): Target node type
        device (torch.device): Device to store graph and tensors on

    Returns:
        tuple: (graph, features, target_id_to_node, id_to_node)
    """
    print(f"Getting relation graphs from edge lists: {edges}")
    edgelists, id_to_node = {}, {}

    for edge in edges:
        edgelist, rev_edgelist, id_to_node, src, dst = parse_edgelist(
            os.path.join(training_dir, edge), id_to_node, header=True
        )

        if src == target_node_type:
            src = "target"
        if dst == target_node_type:
            dst = "target"

        if src == "target" and dst == "target":
            print("Will add self loop for target later...")
        else:
            edgelists[(src, f"{src}<>{dst}", dst)] = edgelist
            edgelists[(dst, f"{dst}<>{src}", src)] = rev_edgelist
            print(
                f"Read edges for {src}<{dst}> from {os.path.join(training_dir, edge)}"
            )

    # Get features for target nodes
    features, new_nodes = get_features(
        id_to_node[target_node_type],
        os.path.join(training_dir, nodes),
        device,
    )
    print("Read features for target nodes")

    # Add self relation
    edgelists[("target", "self_relation", "target")] = [
        (t, t) for t in id_to_node[target_node_type].values()
    ]

    g = dgl.heterograph(edgelists)
    if device is not None:
        g = g.to(device)

    print(
        f"Constructed heterograph with: Node types {g.ntypes}, Edge types {g.canonical_etypes}"
    )
    print(f"Number of target nodes: {g.number_of_nodes('target')}")

    g.nodes["target"].data["features"] = features

    target_id_to_node = id_to_node[target_node_type]
    id_to_node["target"] = target_id_to_node
    del id_to_node[target_node_type]

    return g, features, target_id_to_node, id_to_node


# Set up module logger
logging = get_logger(__name__)
