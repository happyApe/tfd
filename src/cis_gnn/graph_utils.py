import logging
import os
import warnings
from typing import Dict, List, Optional, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class GraphConstructor:
    """Class to handle graph construction and preprocessing."""

    def __init__(
        self,
        feature_scaling: str = "standard",
        add_self_loops: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            feature_scaling: Scaling method ('standard' or 'none')
            add_self_loops: Whether to add self loops to the graph
            cache_dir: Directory to cache processed graphs
        """
        self.feature_scaling = feature_scaling
        self.add_self_loops = add_self_loops
        self.cache_dir = cache_dir
        self.scalers = {}

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def process_features(
        self, features: np.ndarray, node_type: str, is_training: bool = True
    ) -> np.ndarray:
        """Process node features with scaling."""
        if self.feature_scaling == "none":
            return features

        if is_training:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            self.scalers[node_type] = scaler
        else:
            if node_type not in self.scalers:
                warnings.warn(f"No scaler found for {node_type}. Using raw features.")
                return features
            features = self.scalers[node_type].transform(features)

        return features

    def construct_graph(
        self,
        edge_lists: Dict[str, pd.DataFrame],
        node_features: Dict[str, np.ndarray],
        node_labels: Optional[Dict[str, np.ndarray]] = None,
    ) -> dgl.DGLGraph:
        """
        Construct heterogeneous graph from edge lists and features.

        Args:
            edge_lists: Dict of edge lists for each relation type
            node_features: Dict of node features for each node type
            node_labels: Optional dict of node labels

        Returns:
            DGL heterogeneous graph
        """
        # Create node mappings
        node_mappings = self._create_node_mappings(edge_lists, node_features)

        # Process edges
        canonical_etypes = []
        edge_tensors = {}

        for rel_name, edges_df in edge_lists.items():
            src_type, dst_type = self._parse_relation_name(rel_name)

            # Map node IDs to indices
            src_mapped = edges_df.iloc[:, 0].map(node_mappings[src_type])
            dst_mapped = edges_df.iloc[:, 1].map(node_mappings[dst_type])

            # Create edge tensor
            edge_tensor = torch.stack(
                [torch.tensor(src_mapped.values), torch.tensor(dst_mapped.values)]
            )

            canonical_etype = (src_type, rel_name, dst_type)
            canonical_etypes.append(canonical_etype)
            edge_tensors[canonical_etype] = edge_tensor

        # Create graph
        graph_data = {
            etype: (edge_tensors[etype][0], edge_tensors[etype][1])
            for etype in canonical_etypes
        }

        num_nodes_dict = {
            ntype: len(mapping) for ntype, mapping in node_mappings.items()
        }

        g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

        # Add node features
        for ntype, features in node_features.items():
            if features is not None:
                g.nodes[ntype].data["feat"] = torch.tensor(features)

        # Add node labels if provided
        if node_labels:
            for ntype, labels in node_labels.items():
                if labels is not None:
                    g.nodes[ntype].data["label"] = torch.tensor(labels)

        # Add self loops if requested
        if self.add_self_loops:
            for ntype in g.ntypes:
                src_dst_pairs = torch.arange(g.number_of_nodes(ntype))
                src_dst_pairs = torch.stack([src_dst_pairs, src_dst_pairs])
                g.add_edges(
                    src_dst_pairs[0],
                    src_dst_pairs[1],
                    etype=(ntype, f"{ntype}_self", ntype),
                )

        return g

    def _create_node_mappings(
        self, edge_lists: Dict[str, pd.DataFrame], node_features: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        """Create mappings from node IDs to indices."""
        node_mappings = {}

        # Get all unique nodes from edge lists
        for rel_name, edges_df in edge_lists.items():
            src_type, dst_type = self._parse_relation_name(rel_name)

            if src_type not in node_mappings:
                node_mappings[src_type] = {}
            if dst_type not in node_mappings:
                node_mappings[dst_type] = {}

            for node_id in edges_df.iloc[:, 0].unique():
                if node_id not in node_mappings[src_type]:
                    node_mappings[src_type][node_id] = len(node_mappings[src_type])

            for node_id in edges_df.iloc[:, 1].unique():
                if node_id not in node_mappings[dst_type]:
                    node_mappings[dst_type][node_id] = len(node_mappings[dst_type])

        # Add nodes that only appear in features
        for ntype, features in node_features.items():
            if features is not None and ntype not in node_mappings:
                node_mappings[ntype] = {i: i for i in range(len(features))}

        return node_mappings

    @staticmethod
    def _parse_relation_name(rel_name: str) -> Tuple[str, str]:
        """Parse relation name to get source and destination node types."""
        if "<>" in rel_name:
            return rel_name.split("<>")
        return rel_name.split("_to_")

    def save_graph(self, g: dgl.DGLGraph, path: str):
        """Save graph and associated data to disk."""
        if not self.cache_dir:
            raise ValueError("Cache directory not specified")

        save_path = os.path.join(self.cache_dir, path)
        dgl.save_graphs(save_path, [g])

        # Save scalers if any
        if self.scalers:
            scaler_path = os.path.join(self.cache_dir, f"{path}_scalers.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scalers, f)

    def load_graph(self, path: str) -> dgl.DGLGraph:
        """Load graph and associated data from disk."""
        if not self.cache_dir:
            raise ValueError("Cache directory not specified")

        load_path = os.path.join(self.cache_dir, path)
        graphs, _ = dgl.load_graphs(load_path)

        # Load scalers if they exist
        scaler_path = os.path.join(self.cache_dir, f"{path}_scalers.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scalers = pickle.load(f)

        return graphs[0]


def create_masks(
    num_nodes: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create train/val/test masks."""
    np.random.seed(seed)

    indices = np.random.permutation(num_nodes)
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.

    Args:
        labels: Node labels tensor

    Returns:
        Tensor of class weights
    """
    class_counts = torch.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts.float())
    return weights


def add_reverse_edges(g: dgl.DGLGraph) -> dgl.DGLGraph:
    """Add reverse edges to the graph with reversed relation types."""
    new_etypes = []
    new_edges = {}

    for src_type, etype, dst_type in g.canonical_etypes:
        # Skip self-relation edges
        if src_type == dst_type and "self" in etype:
            continue

        # Get original edges
        edges = g.edges(etype=etype)

        # Create reverse relation name
        rev_etype = f"{etype}_rev"
        rev_canonical = (dst_type, rev_etype, src_type)

        # Add to new edge dict
        new_etypes.append(rev_canonical)
        new_edges[rev_canonical] = (edges[1], edges[0])

    # Add new edges to graph
    for new_etype in new_etypes:
        src, dst = new_edges[new_etype]
        g.add_edges(src, dst, etype=new_etype)

    return g


def normalize_adj(g: dgl.DGLGraph) -> dgl.DGLGraph:
    """
    Symmetrically normalize adjacency matrix for each relation type.
    """
    for etype in g.etypes:
        src, _, dst = g.to_canonical_etype(etype)

        # Calculate degree matrices
        src_degrees = g.out_degrees(etype=etype).float()
        dst_degrees = g.in_degrees(etype=etype).float()

        # Add epsilon to avoid division by zero
        src_norm = torch.pow(src_degrees + 1e-6, -0.5)
        dst_norm = torch.pow(dst_degrees + 1e-6, -0.5)

        # Store normalization values
        g.edges[etype].data["norm"] = (
            src_norm[g.edges[etype].data[dgl.EID]]
            * dst_norm[g.edges[etype].data[dgl.EID]]
        )

    return g


def graph_augmentation(g: dgl.DGLGraph, drop_rate: float = 0.1) -> dgl.DGLGraph:
    """
    Augment graph by randomly dropping edges and features.

    Args:
        g: Input graph
        drop_rate: Rate of edges/features to drop

    Returns:
        Augmented graph
    """
    g = g.clone()

    # Edge dropping
    for etype in g.etypes:
        edge_mask = torch.rand(g.number_of_edges(etype)) > drop_rate
        g.edges[etype].data["edge_mask"] = edge_mask

    # Feature masking
    for ntype in g.ntypes:
        if "feat" in g.nodes[ntype].data:
            feat = g.nodes[ntype].data["feat"]
            mask = torch.rand(feat.shape) > drop_rate
            g.nodes[ntype].data["feat"] = feat * mask.float()

    return g


class BatchSampler:
    """Sample subgraphs for mini-batch training."""

    def __init__(
        self,
        g: dgl.DGLGraph,
        batch_size: int,
        num_hops: int = 2,
        node_type: str = "target",
        shuffle: bool = True,
    ):
        self.g = g
        self.batch_size = batch_size
        self.num_hops = num_hops
        self.node_type = node_type
        self.shuffle = shuffle

        self.num_nodes = g.number_of_nodes(node_type)
        self.reset()

    def reset(self):
        """Reset sampler state."""
        self.indices = torch.arange(self.num_nodes)
        if self.shuffle:
            self.indices = self.indices[torch.randperm(self.num_nodes)]
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self) -> dgl.DGLGraph:
        if self.current >= self.num_nodes:
            self.reset()
            raise StopIteration

        # Get batch nodes
        end = min(self.current + self.batch_size, self.num_nodes)
        batch_nodes = self.indices[self.current : end]
        self.current = end

        # Sample subgraph
        sampled_graph = dgl.sampling.sample_neighbors(
            self.g,
            {self.node_type: batch_nodes},
            self.num_hops,
        )

        return sampled_graph

    def __len__(self):
        return (self.num_nodes + self.batch_size - 1) // self.batch_size


# Helper function to get graph statistics
def get_graph_stats(g: dgl.DGLGraph) -> Dict:
    """Get basic statistics of the graph."""
    stats = {
        "num_nodes": {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes},
        "num_edges": {etype: g.number_of_edges(etype) for etype in g.etypes},
        "node_features": {
            ntype: {
                "shape": (
                    tuple(g.nodes[ntype].data["feat"].shape)
                    if "feat" in g.nodes[ntype].data
                    else None
                )
            }
            for ntype in g.ntypes
        },
    }

    # Add average degree statistics
    for ntype in g.ntypes:
        in_degrees = []
        out_degrees = []

        for etype in g.etypes:
            src, _, dst = g.to_canonical_etype(etype)
            if src == ntype:
                out_degrees.extend(g.out_degrees(etype=etype).tolist())
            if dst == ntype:
                in_degrees.extend(g.in_degrees(etype=etype).tolist())

        stats[f"{ntype}_avg_in_degree"] = np.mean(in_degrees) if in_degrees else 0
        stats[f"{ntype}_avg_out_degree"] = np.mean(out_degrees) if out_degrees else 0

    return stats
