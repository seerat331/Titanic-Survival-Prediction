import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from src.logger import logger
from src.exception import ModelTrainingError

print("Running model_training.py")
print("Logger object:", logger)


class ModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(),
        }

    def train_models(self, X_train, y_train):
        trained_models = {}
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}")
                model.fit(X_train, y_train)
                trained_models[name] = model
                logger.info(f"{name} trained successfully")
            except Exception as e:

                logger.exception(f"Error while training {name}")

                raise ModelTrainingError(f"Training failed for {name}: {e}")

        logger.info("All models trained successfully")
        return trained_models

    def save_model(self, model, path):
        joblib.dump(model, path)
        logger.info(f"Model saved at {path}")
