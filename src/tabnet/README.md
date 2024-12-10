# Credit Card Fraud Detection for Capital One Synthetic Dataset

This part of project implements credit card fraud detection using two different models:
1. Logistic Regression
2. TabNet (a deep learning approach)

## Dataset

The dataset used in this project is available in the Capital One Recruiting GitHub repository:

Get the dataset by running these commands:
```bash
# Clone the repository
git clone https://github.com/CapitalOneRecruiting/DS.git

# Navigate into the repository
cd DS

# Unzip the transactions file
unzip transactions.zip

# Move the transactions.txt file to your project directory
# Assuming your project directory is one level up
mv transactions.txt ../

# Return to your project directory
cd ..
```

The dataset contains credit card transactions with various features including transaction amount, merchant information, and fraud labels.

## Project Structure

```
├── data_processor.py     # Data loading and preprocessing
├── models.py            # Model implementations
├── trainer.py           # Training and evaluation utilities
├── main.py             # Main execution script
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script with desired arguments:

```bash
# Train TabNet model
python main.py --model_type tabnet --data_path path/to/transactions.txt --epochs 30 --batch_size 128 --learning_rate 0.001

# Train Logistic Regression model
python main.py --model_type logistic --data_path path/to/transactions.txt
```

### Command Line Arguments

- `--data_path`: Path to the transactions data file (default: 'transactions.txt')
- `--model_type`: Type of model to train ['tabnet', 'logistic'] (default: 'tabnet')
- `--batch_size`: Batch size for training (default: 128)
- `--epochs`: Number of epochs to train (default: 30)
- `--learning_rate`: Learning rate (default: 0.001)

## Models

### Logistic Regression
A simple baseline model using scikit-learn's implementation with SMOTE for handling class imbalance.

### TabNet
A deep learning model that uses attention mechanisms and feature selection, implemented in PyTorch. The model includes:
- Feature transformers with batch normalization and dropout
- Attentive transformers
- Focal loss for handling class imbalance

## Data Processing

The data processor handles:
- Loading transaction data
- Feature engineering
- Categorical encoding
- Data scaling
- Train/validation/test splitting
- SMOTE resampling for handling class imbalance

## Evaluation

Models are evaluated using:
- Classification report (precision, recall, F1-score)
- ROC-AUC score
- Confusion matrix
- Precision-Recall curve

For TabNet, training curves (train/validation loss) are also plotted.
