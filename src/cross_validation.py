from sklearn.model_selection import cross_val_score


class CrossValidation:
    def evaluate(self, model, X_train, y_train):
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        print("\nCross Validation Scores")
        print(scores)
        print(f"\nMean Accuracy:{scores.mean():4f}")
        print(f"Standard Deviation:{scores.std():.4f}")
        return scores
