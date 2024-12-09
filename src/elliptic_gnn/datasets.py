import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


class EllipticDataset:
    def __init__(
        self,
        features_path="data/elliptic_bitcoin_dataset/elliptic_txs_features.csv",
        edges_path="data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv",
        classes_path="data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv",
    ):
        """Initialize the Elliptic Dataset.

        Args:
            features_path (str): Path to features CSV file
            edges_path (str): Path to edges CSV file
            classes_path (str): Path to classes CSV file
        """
        self.features_df = pd.read_csv(features_path, header=None)
        self.edges_df = pd.read_csv(edges_path)
        self.labels_df = pd.read_csv(classes_path)
        self.labels_df["class"] = self.labels_df["class"].map(
            {"unknown": 2, "1": 1, "2": 0}
        )
        self.merged_df = self.merge()
        self.edge_index = self._edge_index()
        self.edge_weights = self._edge_weights()
        self.node_features = self._node_features()
        self.labels = self._labels()
        self.classified_ids = self._classified_ids()
        self.unclassified_ids = self._unclassified_ids()
        self.licit_ids = self._licit_ids()
        self.illicit_ids = self._illicit_ids()

    def visualize_distribution(self, save_path=None):
        """Visualize the distribution of classes."""
        groups = self.labels_df.groupby("class").count()
        plt.figure(figsize=(10, 6))
        plt.title("Classes distribution")
        plt.barh(
            ["Licit", "Illicit", "Unknown"],
            groups["txId"].values,
            color=["green", "red", "grey"],
        )
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def merge(self):
        """Merge features with labels."""
        df_merge = self.features_df.merge(
            self.labels_df, how="left", right_on="txId", left_on=0
        )
        df_merge = df_merge.sort_values(0).reset_index(drop=True)
        return df_merge

    def train_test_split(self, test_size=0.15, random_state=42):
        """Split the dataset into train and validation sets."""
        train_idx, valid_idx = train_test_split(
            self.classified_ids.values,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels[self.classified_ids],
        )
        return train_idx, valid_idx

    def pyg_dataset(self, test_size=0.15, random_state=42):
        """Convert to PyTorch Geometric dataset format."""
        dataset = Data(
            x=self.node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_weights,
            y=self.labels,
        )
        train_idx, valid_idx = self.train_test_split(
            test_size=test_size, random_state=random_state
        )
        dataset.train_idx = train_idx
        dataset.valid_idx = valid_idx
        dataset.test_idx = self.unclassified_ids

        return dataset

    def _licit_ids(self):
        """Get indices of licit transactions."""
        node_features = self.merged_df.drop(["txId"], axis=1).copy()
        return node_features["class"].loc[node_features["class"] == 0].index

    def _illicit_ids(self):
        """Get indices of illicit transactions."""
        node_features = self.merged_df.drop(["txId"], axis=1).copy()
        return node_features["class"].loc[node_features["class"] == 1].index

    def _classified_ids(self):
        """Get indices of labeled transactions."""
        node_features = self.merged_df.drop(["txId"], axis=1).copy()
        return node_features["class"].loc[node_features["class"] != 2].index

    def _unclassified_ids(self):
        """Get indices of unlabeled transactions."""
        node_features = self.merged_df.drop(["txId"], axis=1).copy()
        return node_features["class"].loc[node_features["class"] == 2].index

    def _node_features(self):
        """Extract node features."""
        node_features = self.merged_df.drop(["txId"], axis=1).copy()
        node_features = node_features.drop(columns=[0, 1, "class"])
        return torch.tensor(node_features.values, dtype=torch.float64)

    def _edge_index(self):
        """Create edge index tensor."""
        node_ids = self.merged_df[0].values
        ids_mapping = {y: x for x, y in enumerate(node_ids)}
        edges = self.edges_df.copy()
        edges.txId1 = edges.txId1.map(ids_mapping)
        edges.txId2 = edges.txId2.map(ids_mapping)
        edges = edges.astype(int)

        edge_index = np.array(edges.values).T
        return torch.tensor(edge_index, dtype=torch.long).contiguous()

    def _edge_weights(self):
        """Create edge weights tensor."""
        return torch.tensor([1] * self.edge_index.shape[1], dtype=torch.float64)

    def _labels(self):
        """Create labels tensor."""
        labels = self.merged_df["class"].values
        return torch.tensor(labels, dtype=torch.float64)
