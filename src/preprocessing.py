import pandas as pd


class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def show_shape(self):
        print("=" * 60)
        print("Dataset shape")
        print("=" * 60)
        print(f"Rows :{self.df.shape[0]}")
        print(f"Columns :{self.df.shape[1]}")

    def show_head(self, rows=5):
        print("=" * 60)
        print("First Rows")
        print("=" * 60)
        print(self.df.head(rows))

    def show_tail(self, rows=5):
        print("=" * 60)
        print("Last Rows")
        print("=" * 60)
        print(self.df.tail(rows))

    def dataset_info(self):
        print("=" * 60)
        print("Dataset Information")
        print("=" * 60)
        self.df.info()

    def missing_values(self):
        print("=" * 60)
        print("Missing Values in Dataset")
        print("=" * 60)
        print(self.df.isnull().sum())

    def duplicate_values(self):
        print("=" * 60)
        print("Duplicate Values")
        print("=" * 60)
        print(self.df.duplicated().sum())

    def statistical_summary(self):
        print("=" * 60)
        print("Statistical Summary")
        print("=" * 60)
        print(self.df.describe(include="all"))

    def dataset_datatypes(self):
        print("=" * 60)
        print("DataTypes")
        print("=" * 60)
        print(self.df.dtypes)

    def unique_values(self):
        print("=" * 60)
        print("Unique Values")
        print("=" * 60)

        for column in self.df.columns:
            print(f"{column}:{self.df[column].nunique()}")

    def full_report(self):
        self.show_shape()
        print()

        self.show_head()
        print()

        self.dataset_info()
        print()

        self.show_tail()
        print()

        self.dataset_datatypes()
        print()

        self.duplicate_values()
        print()

        self.missing_values()
        print()

        self.unique_values()
        print()

        self.statistical_summary()
