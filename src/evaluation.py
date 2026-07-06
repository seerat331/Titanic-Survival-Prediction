from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

class Evaluator:
    def evaluate_models(self, models, X_test, y_test):
        results={}


        for name, model in models.items():
            
            predictions=model.predict(X_test)
            accuracy=accuracy_score(
                y_test, predictions
            )
            precision=precision_score(
                y_test, predictions
            )
            recall=recall_score(
                y_test, predictions
            )
            f1=f1_score(
                y_test, predictions
            )

            results[name]={
                "Accuracy":accuracy,
                "Precision":precision,
                "Recall":recall,
                "F1 Score":f1
            }
        return results
        