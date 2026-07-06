import joblib 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

class ModelTrainer:
    def __init__(self):
        self.models={
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "SVM":SVC()

        }
    
    def train_models(self, X_train, y_train):
        trained_models={}
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name]=model
        print("\nAll models trained successfully!")
        return  trained_models
    
    def save_model(self, model, path):
        joblib.dump(model, path)
        print(f"\nModel saved at:\n{path}")