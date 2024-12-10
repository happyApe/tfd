import datetime
import json

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataProcessor:
    def __init__(self):
        self.categorical_columns = [
            "merchantName",
            "acqCountry",
            "merchantCountryCode",
            "merchantCategoryCode",
            "merchantCity",
            "merchantState",
            "merchantZip",
        ]
        self.label_encoders = {}

    def load_transactions(self, path: str) -> pd.DataFrame:
        """Load transactions from text file."""
        with open(path) as f:
            lines = [json.loads(x) for x in f.readlines()]
        return pd.DataFrame(lines)

    def process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert date strings to datetime objects."""
        date_columns = [
            "transactionDateTime",
            "accountOpenDate",
            "dateOfLastAddressChange",
        ]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables using LabelEncoder."""
        for col in self.categorical_columns:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features."""
        # Credit utilization features
        df["credit_utilization"] = df["currentBalance"] / df["creditLimit"]
        df["tra_to_limit"] = df["transactionAmount"] / df["creditLimit"]
        df["cb_to_limit"] = df["currentBalance"] / df["creditLimit"]

        # Time-based features
        df["account_age"] = (
            df["transactionDateTime"].dt.date - df["accountOpenDate"].dt.date
        ).apply(lambda x: x.days)
        df["address_change_age"] = (
            df["transactionDateTime"].dt.date - df["dateOfLastAddressChange"].dt.date
        ).apply(lambda x: x.days)

        # CVV match feature
        df["isCVVEqual"] = np.where(df["cardCVV"] == df["enteredCVV"], 1, 0)

        # Aggregated features
        df["customerTotalTransactions"] = df.groupby("customerId")[
            "transactionAmount"
        ].transform("count")
        df["customerAvgTransactionAmount"] = df.groupby("customerId")[
            "transactionAmount"
        ].transform("mean")
        df["merchantTotalTransactions"] = df.groupby("merchantName")[
            "transactionAmount"
        ].transform("count")
        df["merchantAvgTransactionAmount"] = df.groupby("merchantName")[
            "transactionAmount"
        ].transform("mean")

        return df

    def get_features(self, df: pd.DataFrame) -> tuple:
        """Get feature matrix X and target vector y."""
        features = [
            "availableMoney",
            "transactionAmount",
            "creditLimit",
            "credit_utilization",
            "tra_to_limit",
            "cb_to_limit",
            "account_age",
            "address_change_age",
            "customerTotalTransactions",
            "customerAvgTransactionAmount",
            "merchantTotalTransactions",
            "merchantAvgTransactionAmount",
            "isCVVEqual",
        ] + self.categorical_columns

        X = df[features]
        y = df["isFraud"]
        return X, y

    def prepare_data(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.3
    ) -> tuple:
        """Prepare train and test sets with SMOTE sampling."""
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Apply SMOTE to training data
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def process_pipeline(self, data_path: str) -> tuple:
        """Complete data processing pipeline."""
        df = self.load_transactions(data_path)
        df = self.process_dates(df)
        df = self.encode_categorical(df)
        df = self.engineer_features(df)
        X, y = self.get_features(df)
        return self.prepare_data(X, y)
