import argparse

import torch
from data_processor import DataProcessor
from models import LogisticModel, TabNetFraudDetector
from trainer import ModelTrainer


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection")
    parser.add_argument(
        "--data_path",
        type=str,
        default="transactions.txt",
        help="Path to transactions data file",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="tabnet",
        choices=["tabnet", "logistic"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize processor and load data
    print("Processing data...")
    processor = DataProcessor()
    X_train, X_val, X_test, y_train, y_val, y_test = processor.process_pipeline(
        args.data_path
    )

    # Initialize trainer
    trainer = ModelTrainer(device=device)

    if args.model_type == "tabnet":
        print("Training TabNet model...")
        # Initialize TabNet model
        model = TabNetFraudDetector(input_dim=X_train.shape[1], dropout_rate=0.1).to(
            device
        )

        # Train model
        model, train_losses, val_losses = trainer.train_tabnet(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
        )

        # Plot training curves
        trainer.plot_training_curves(train_losses, val_losses)

        # Evaluate model
        print("\nEvaluating TabNet model on test set:")
        trainer.evaluate_model(model, X_test, y_test, model_type="tabnet")

    else:  # logistic regression
        print("Training Logistic Regression model...")
        # Initialize and train logistic regression
        log_model = LogisticModel()
        log_model.fit(X_train, y_train)

        # Evaluate model
        print("\nEvaluating Logistic Regression model on test set:")
        trainer.evaluate_model(log_model, X_test, y_test, model_type="logistic")


if __name__ == "__main__":
    main()
