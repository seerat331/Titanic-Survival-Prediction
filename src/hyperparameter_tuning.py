from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class HyperParameterTuner:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.parameters = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

    def tune(self, X_train, y_train):
        print("\nRunning GridSearchCV...\n")
        grid = GridSearchCV(
            estimator=self.model,
            param_grid=self.parameters,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        print("Grid Search Completed!\n")
        print("Best Parameters")
        print(grid.best_params_)
        print()
        print("Best Accuracy")
        print(grid.best_score_)
        return grid.best_estimator_
