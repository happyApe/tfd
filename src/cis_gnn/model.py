import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Multi-head attention layer for heterogeneous graphs."""

    def __init__(self, in_size, out_size, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.out_size = out_size
        head_size = out_size // num_heads
        assert (
            head_size * num_heads == out_size
        ), "out_size must be divisible by num_heads"

        self.key = nn.Linear(in_size, out_size)
        self.query = nn.Linear(in_size, out_size)
        self.value = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        k = self.key(h).view(-1, self.num_heads, self.out_size // self.num_heads)
        q = self.query(h).view(-1, self.num_heads, self.out_size // self.num_heads)
        v = self.value(h).view(-1, self.num_heads, self.out_size // self.num_heads)

        # Scaled dot-product attention
        attn = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(
            self.out_size // self.num_heads
        )
        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v)
        return out.view(-1, self.out_size)


class HeteroRGCNLayer(nn.Module):
    """Improved heterogeneous GCN layer with attention and residual connections."""

    def __init__(
        self, in_size, out_size, etypes, num_heads=4, dropout=0.2, use_attention=True
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.use_attention = use_attention

        # Transform weights for each relation
        self.weight = nn.ModuleDict(
            {name: nn.Linear(in_size, out_size) for name in etypes}
        )

        # Attention layers
        if use_attention:
            self.attention = nn.ModuleDict(
                {
                    name: AttentionLayer(out_size, out_size, num_heads, dropout)
                    for name in etypes
                }
            )

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(out_size)
        self.dropout = nn.Dropout(dropout)

        # Residual connection transform if needed
        self.residual = nn.Linear(in_size, out_size) if in_size != out_size else None

    def forward(self, G, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            if srctype in feat_dict:
                # Transform features
                Wh = self.weight[etype](feat_dict[srctype])

                # Apply attention if enabled
                if self.use_attention:
                    Wh = self.attention[etype](Wh)

                # Store in graph for message passing
                G.nodes[srctype].data[f"Wh_{etype}"] = Wh
                funcs[etype] = (fn.copy_u(f"Wh_{etype}", "m"), fn.mean("m", "h"))

        # Message passing
        G.multi_update_all(funcs, "sum")

        # Get updated features
        out_dict = {}
        for ntype in G.ntypes:
            if "h" in G.nodes[ntype].data:
                # Get node features
                h = G.nodes[ntype].data["h"]

                # Apply residual connection if possible
                if self.residual is not None and ntype in feat_dict:
                    h = h + self.residual(feat_dict[ntype])
                elif ntype in feat_dict and h.shape == feat_dict[ntype].shape:
                    h = h + feat_dict[ntype]

                # Apply normalization and dropout
                h = self.layer_norm(h)
                h = self.dropout(h)

                out_dict[ntype] = h

        return out_dict


class HeteroRGCN(nn.Module):
    """Improved heterogeneous GCN with attention and skip connections."""

    def __init__(
        self,
        ntype_dict,
        etypes,
        in_size,
        hidden_size,
        out_size,
        n_layers,
        embedding_size,
        num_heads=4,
        dropout=0.2,
        use_attention=True,
    ):
        super().__init__()

        # Create trainable embeddings for non-target nodes
        self.embed = nn.ParameterDict(
            {
                ntype: nn.Parameter(torch.Tensor(num_nodes, in_size))
                for ntype, num_nodes in ntype_dict.items()
                if ntype != "target"
            }
        )

        # Initialize embeddings with Xavier uniform
        for embed in self.embed.values():
            nn.init.xavier_uniform_(embed)

        # Create RGCN layers with residual connections
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            HeteroRGCNLayer(
                embedding_size, hidden_size, etypes, num_heads, dropout, use_attention
            )
        )

        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(
                HeteroRGCNLayer(
                    hidden_size, hidden_size, etypes, num_heads, dropout, use_attention
                )
            )

        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_size),
        )

        self.loss_fn = FocalLoss(alpha=0.25, gamma=2)

    def forward(self, g, features):
        # Get embeddings for all node types
        h_dict = {ntype: emb for ntype, emb in self.embed.items()}
        h_dict["target"] = features

        # Forward pass through RGCN layers
        for i, layer in enumerate(self.layers):
            h_dict = layer(g, h_dict)
            if i != len(self.layers) - 1:
                # Apply non-linearity between layers
                h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}

        # Final classification
        return self.classifier(h_dict["target"])

    def get_loss(self, logits, labels):
        return self.loss_fn(logits, labels)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
