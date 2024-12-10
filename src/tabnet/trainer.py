import matplotlib.pyplot as plt
import numpy as np
import torch
from models import FocalLoss
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class ModelTrainer:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def train_tabnet(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=128,
        num_epochs=30,
        learning_rate=0.001,
    ):
        # Prepare data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.FloatTensor(y_train).to(self.device),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val).to(self.device),
            torch.FloatTensor(y_val).to(self.device),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize optimizer and loss
        criterion = FocalLoss(alpha=0.25, gamma=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )

        best_val_loss = float("inf")
        train_losses = {}
        val_losses = {}

        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_X, batch_y in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"
            ):
                optimizer.zero_grad()
                y_pred, M_loss = model(batch_X)
                loss = criterion(y_pred, batch_y.view(-1, 1)) + 0.001 * M_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    y_pred, M_loss = model(batch_X)
                    loss = criterion(y_pred, batch_y.view(-1, 1)) + 0.001 * M_loss
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_losses[epoch] = train_loss
            val_losses[epoch] = val_loss

            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_tabnet_model.pth")

        return model, train_losses, val_losses

    def evaluate_model(self, model, X_test, y_test, model_type="tabnet"):
        if model_type == "tabnet":
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
                predictions, _ = model(X_test_tensor)
                predictions = predictions.cpu().numpy()
        else:  # logistic
            predictions = model.predict_proba(X_test)[:, 1]

        pred_classes = (predictions > 0.5).astype(int)

        # Print metrics
        print("\nClassification Report:")
        print(classification_report(y_test, pred_classes))

        print("\nROC AUC Score:")
        print(roc_auc_score(y_test, predictions))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, pred_classes))

        # Plot Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, predictions)
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.show()

    def plot_training_curves(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(
            list(train_losses.keys()), list(train_losses.values()), label="Train Loss"
        )
        plt.plot(
            list(val_losses.keys()), list(val_losses.values()), label="Validation Loss"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()
