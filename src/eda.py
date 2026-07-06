import matplotlib.pyplot as plt
import seaborn as sns
import os
class EDA:
    def __init__(self, df, figure_dir):
        self.df=df
        self.figure_dir=figure_dir

        sns.set_style("whitegrid")

    def save_plot(self, filename):
        path=os.path.join(self.figure_dir, filename) 
        plt.savefig(path, dpi=300,bbox_inches="tight")
        plt.close()

    def survival_count(self):
        plt.figure(figsize=(6,4))
        sns.countplot(
            data=self.df,
            x="Survived"

        )
        plt.title("Survial Counts")
        self.save_plot("survival_count.png")

    def passenger_class(self):
        plt.figure(figsize=(6,4))
        sns.countplot(
            data=self.df,
            x="Pclass"
        )
        plt.title("Passenger Class")
        self.save_plot("passenger_class.png")

    def gender_distribution(self):
        plt.figure(figsize=(6,4))
        sns.countplot(
            data=self.df,
            x="Sex"

        )
        plt.title("Gender Distribution")
        self.save_plot("gender_distribution.png")

    def age_distribution(self):
        plt.figure(figsize=(8,5))
        sns.histplot(
            self.df["Age"],
            bins=30,
            kde=True
        )
        plt.title("Age Distribution")
        self.save_plot("age_distribution.png")

    def fare_distribution(self):
        plt.figure(figsize=(8,5))
        sns.histplot(
            self.df["Fare"],
            bins=30,
            kde=True
        )
        plt.title("Fare Distribution")
        self.save_plot("fare_distribution.png")

    def survival_by_gender(self):
        plt.figure(figsize=(6,4))
        sns.countplot(
            data=self.df,
            x="Sex",
            hue="Survived"
        )
        plt.title("Survival by Gender")
        self.save_plot("survival_by_gender.png")

    def survival_by_class(self):
        plt.figure(figsize=(6,4))
        sns.countplot(
            data=self.df,
            x="Pclass",
            hue="Survived"
        )
        plt.title("Survival by Passenger Class")
        self.save_plot("survival_by_class.png")

    def correlation_heatmap(self):
        plt.figure(figsize=(10,8))
        correlation=self.df.select_dtypes(include="number").corr()
        sns.heatmap(
            correlation,
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )
        plt.title("Correlation Heatmap")
        self.save_plot("correlation_heatmap.png")

    def run_all(self):

        print("\nGenerating EDA plts...\n")
        self.survival_count()
        self.passenger_class()
        self.age_distribution()
        self.gender_distribution()
        self.survival_by_class()
        self.survival_by_gender()
        self.correlation_heatmap()
        self.fare_distribution()
        print("All plots saved successfully!")