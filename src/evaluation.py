import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)


class Evaluator:
    def evaluate_models(self, models, X_test, y_test):
        results = {}

        for name, model in models.items():

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)

            results[name] = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
            }
        return results

    # Classification report
    def classification_report(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        print("\nClassification REport\n")
        print(classification_report(y_test, predictions))

    # Confusion matrix
    def confusion_matrux_plot(self, model, X_test, y_test, save_path):
        predications = model.predict(X_test)
        cm = confusion_matrix(y_test, predications)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.savefig(save_path)
        plt.close()

    # ROC Curve
    def roc_curve_plot(self, model, X_test, y_test, save_path):
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title("ROC Curve")
        plt.savefig(save_path)
        plt.close()

    # Save Metrics
    def save_results(self, results, save_path):
        df = pd.DataFrame(results).T
        df.to_csv(save_path)
        print(f"\nMetrics saved to:\n{save_path}")
