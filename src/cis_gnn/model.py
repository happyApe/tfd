import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroRGCNLayer(nn.Module):
    """
    Heterogeneous GCN layer that handles multiple relation types.
    """

    def __init__(self, in_size, out_size, etypes, device=None):
        """
        Args:
            in_size (int): Input feature size
            out_size (int): Output feature size
            etypes (list): List of edge types in the graph
            device (torch.device): Device to place the layer on
        """
        super(HeteroRGCNLayer, self).__init__()
        # Create linear transformation for each relation
        self.weight = nn.ModuleDict(
            {name: nn.Linear(in_size, out_size) for name in etypes}
        )

        if device is not None:
            self.to(device)

    def forward(self, G, feat_dict):
        """
        Args:
            G (DGLGraph): The graph
            feat_dict (dict): Node features for each node type

        Returns:
            dict: Updated node features
        """
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            if srctype in feat_dict:
                # Transform features
                Wh = self.weight[etype](feat_dict[srctype])
                # Store in graph for message passing
                G.nodes[srctype].data[f"Wh_{etype}"] = Wh
                # Define message and reduce functions
                funcs[etype] = (fn.copy_u(f"Wh_{etype}", "m"), fn.mean("m", "h"))

        # Perform message passing
        G.multi_update_all(funcs, "sum")

        # Return updated features
        return {
            ntype: G.nodes[ntype].data["h"]
            for ntype in G.ntypes
            if "h" in G.nodes[ntype].data
        }


class HeteroRGCN(nn.Module):
    """
    Heterogeneous Graph Convolutional Network.
    """

    def __init__(
        self,
        ntype_dict,
        etypes,
        in_size,
        hidden_size,
        out_size,
        n_layers,
        embedding_size,
        device=None,
    ):
        """
        Args:
            ntype_dict (dict): Number of nodes for each node type
            etypes (list): Edge types in the graph
            in_size (int): Input feature size
            hidden_size (int): Hidden layer size
            out_size (int): Output size (num classes)
            n_layers (int): Number of RGCN layers
            embedding_size (int): Size of learned node embeddings
            device (torch.device): Device to place the model on
        """
        super(HeteroRGCN, self).__init__()

        # Create trainable embeddings for non-target nodes
        self.embed = nn.ParameterDict(
            {
                ntype: nn.Parameter(torch.Tensor(num_nodes, in_size))
                for ntype, num_nodes in ntype_dict.items()
                if ntype != "target"
            }
        )

        # Initialize embeddings
        for embed in self.embed.values():
            nn.init.xavier_uniform_(embed)

        # Create RGCN layers
        self.layers = nn.ModuleList()
        self.layers.append(HeteroRGCNLayer(embedding_size, hidden_size, etypes, device))

        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(
                HeteroRGCNLayer(hidden_size, hidden_size, etypes, device)
            )

        # Output classification layer
        self.classifier = nn.Linear(hidden_size, out_size)

        if device is not None:
            self.to(device)

    def forward(self, g, features):
        """
        Args:
            g (DGLGraph): The input graph
            features (torch.Tensor): Input features for target nodes

        Returns:
            torch.Tensor: Logits for node classification
        """
        # Get embeddings for all node types
        h_dict = {ntype: emb for ntype, emb in self.embed.items()}
        h_dict["target"] = features

        # Forward pass through RGCN layers
        for i, layer in enumerate(self.layers):
            if i != 0:
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h_dict = layer(g, h_dict)

        # Final classification layer
        return self.classifier(h_dict["target"])
