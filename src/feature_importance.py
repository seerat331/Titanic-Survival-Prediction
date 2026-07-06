import pandas as pd
import matplotlib.pyplot as plt
class FeatureImportance:
    def plot(self, model, X_train, save_path):
        if not hasattr(model, "feature_importances_"):
            print("This model does not support feature importance.")
            return
        importance=pd.DataFrame(
            {
                "Feature":X_train.columns, 
                "Importance":model.feature_importances_
            }
        )
        importance=importance.sort_values(
            by="Importance",
            ascending=False
        )
        print("\nFeature Importance\n")
        print(importance)
        plt.figure(figsize=(8,6))
        plt.barh(
            importance["Feature"],
            importance["Importance"]
        )        
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()