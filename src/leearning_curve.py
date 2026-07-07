import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

class LeaarningCurve:
    def plot(self, model, X_train, y_train, save_path):
        train_sizes, train_scores, validation_scores=learning_curve(
            estimator=model,
            X=X_train, 
            y=y_train,
            cv=5,
            scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1
        )
        train_mean=train_scores.mean(axis=1)
        validation_mean=validation_scores.mean(axis=1)

        plt.figure(figsize=(8,6))

        plt.plot(
            train_mean,
            train_scores,
            marker="o",
            label="Training Accuracy"
        )
        plt.plot(
            train_sizes,
            validation_mean,
            marker="o",
            label="Validation Accuracy"
        )
        plt.title("Learning Curve")
        plt.xlabel("Training Examples")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"\nLearning curve saved to:\n{save_path}")