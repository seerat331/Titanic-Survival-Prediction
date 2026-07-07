import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class FeatureEngineering:
    def __init__(self, df, target_column):
        self.df = df.copy()
        self.target_column = target_column

    def handle_missing_values(self):
        self.df["Age"] = self.df["Age"].fillna(self.df["Age"].median())
        self.df["Embarked"] = self.df["Embarked"].fillna(self.df["Embarked"].mode()[0])
        if "Cabin" in self.df.columns:
            self.df.drop(columns=["Cabin"], inplace=True)

    def drop_columns(self):
        columns = ["PassengerId", "Name", "Ticket"]
        self.df.drop(columns=columns, inplace=True)

    def encode_features(self):
        encoder = LabelEncoder()
        self.df["Sex"] = encoder.fit_transform(self.df["Sex"])
        self.df["Embarked"] = encoder.fit_transform(self.df["Embarked"])

    def split_dataset(self, test_size=0.2, random_state=42):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def save_processed_dataset(self, path):
        self.df.to_csv(path, index=False)
        print(f"Processed dataset saved to:\n{path}")

    def process(self):
        self.handle_missing_values()
        self.drop_columns()
        self.encode_features()
        return self.df
