# | Transaction Fraud Detection with GNNs

Implementation of Graph Neural Networks for fraud detection on the Elliptic Bitcoin and IEEE-CIS datasets.

## Project Structure
```
transaction_fraud_detection_with_gnns_tabnet/
├── src/
│   ├── cis_gnn/               # IEEE-CIS implementation
│   │   ├── data/
│   │   ├── graph_utils.py
│   │   ├── model.py
│   │   ├── process_data.py
│   │   ├── train.py
│   │   └── utils.py
│   │
│   |── elliptic_gnn/         # Elliptic Bitcoin implementation
│   |   ├── datasets.py
│   |   ├── models.py
│   |   ├── trainer.py
│   |   └── main.py
│   └── tabnet/               # TabNet implementation
|        ├── data_processor.py     
|        ├── models.py            
|        ├── trainer.py          
|        ├── main.py            
├── README.md
└── requirements.txt
```

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```
As I was using CUDA 12.4, For DGL compatibility, I had to install the following:
```bash
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```

2. Download Elliptic Dataset:
```bash
pip3 install kaggle
kaggle datasets download -d ellipticco/elliptic-data-set
unzip elliptic-data-set.zip -d src/elliptic_gnn/data/
```

3. Download IEEE-CIS Dataset:
```bash
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip
mkdir -p src/cis_gnn/data/ieee_cis
mv *.csv src/cis_gnn/data/ieee_cis/
```

## Usage

### Elliptic Bitcoin Dataset
```bash
cd src/elliptic_gnn
python main.py --model-type gat --epochs 100
```

### IEEE-CIS Dataset
```bash
cd src/cis_gnn
python process_data.py
python train.py
```

## Models

- GAT (Graph Attention Network)
- GCN (Graph Convolutional Network)
- GIN (Graph Isomorphism Network)

For detailed implementation and parameters, refer to respective source files.

## Results

### Elliptic Bitcoin Dataset

| Model | Accuracy | F1-Macro | Precision | Recall |
| --- | --- | --- | --- | --- |
| GAT | 0.994 | 0.983 | 0.979 | 0.960 |
| GCN | 0.976 | 0.925 | 0.939 | 0.739 |
| GIN | 0.981 | 0.947 | 0.931 | 0.879 |

### IEEE-CIS Dataset

| Metric | Value |
| --- | --- |
| F1 Score | 0.0665 |
| Precision | 0.0344 |
| Recall | 1.0000 |
| Accuracy | 0.0344 |
| ROC AUC | 0.9125 |
| PR AUC | 0.5781 |
| Average Precision | 0.5781 |

### Capital One Synthetic Dataset

| Metric | Value |
| ------ | ----- |
| F1 Score (True) | 0.07 |
| F1 Score (False) | 0.85 |
| Precision (False) | 0.99 |
| Precision (True) | 0.04 |
| Recall (False) | 0.75 |
| Recall (True) | 0.67 |
| Accuracy | 0.75 |
| Macro Avg F1 Score | 0.46 |
| Weighted Avg F1 Score | 0.84 |
| ROC AUC Score | 0.7824 |
