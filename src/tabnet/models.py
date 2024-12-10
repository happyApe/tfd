import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


class TabNetFraudDetector(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=1,
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.3,
        epsilon=1e-15,
        dropout_rate=0.1,
    ):
        super(TabNetFraudDetector, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.dropout_rate = dropout_rate

        self.feature_transformer = nn.ModuleList()
        self.attentive_transformer = nn.ModuleList()

        for step in range(n_steps):
            self.feature_transformer.append(
                nn.Sequential(
                    nn.Linear(input_dim, n_d + n_a),
                    nn.BatchNorm1d(n_d + n_a),
                    nn.ReLU(inplace=False),
                    nn.Dropout(dropout_rate),
                    nn.Linear(n_d + n_a, n_d + n_a),
                    nn.BatchNorm1d(n_d + n_a),
                    nn.ReLU(inplace=False),
                    nn.Dropout(dropout_rate),
                )
            )
            self.attentive_transformer.append(
                nn.Sequential(
                    nn.Linear(input_dim, n_a),
                    nn.BatchNorm1d(n_a),
                    nn.ReLU(inplace=False),
                    nn.Dropout(dropout_rate),
                    nn.Linear(n_a, input_dim),
                    nn.Sigmoid(),
                )
            )

        self.final_layer = nn.Sequential(
            nn.Linear(n_d, n_d // 2),
            nn.BatchNorm1d(n_d // 2),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout_rate),
            nn.Linear(n_d // 2, output_dim),
        )

    def forward(self, x):
        prior = torch.ones(x.shape).to(x.device)
        M_loss = 0

        for step in range(self.n_steps):
            M = self.attentive_transformer[step](prior * x)
            M_loss += torch.mean(torch.sum(torch.abs(M[:, 1:] - M[:, :-1]), dim=1))

            masked_x = M * x
            out = self.feature_transformer[step](masked_x)
            d = F.relu(out[:, : self.n_d], inplace=False)

            if step == 0:
                aggregated_d = d
            else:
                aggregated_d = aggregated_d + d

            prior = torch.mul(self.gamma - M, prior)

        final_out = self.final_layer(aggregated_d)
        return torch.sigmoid(final_out), M_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)


class LogisticModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
