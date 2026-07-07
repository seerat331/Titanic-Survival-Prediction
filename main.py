from src.utils import create_directories
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.eda import EDA
from src.evaluation import Evaluator
from src.model_training import ModelTrainer
from src.feature_importance import FeatureImportance
from src.hyperparameter_tuning import HyperParameterTuner
from src.feature_engineering import FeatureEngineering
from src.leearning_curve import LeaarningCurve
from src.predictor import Predictor
from src.cross_validation import CrossValidation
from src.config import (
    RAW_DATA_PATH,
    MODEL_DIR,
    FIGURE_DIR,
    REPORT_DIR
)
from src.config import (
    RAW_DATA_PATH,
    MODEL_DIR,
    FIGURE_DIR,
    TARGET_COLUMN,
    RANDOM_STATE,
    REPORT_DIR,
    TEST_SIZE,
    PROCESSED_DATA_PATH
)

def main():
    create_directories(
        MODEL_DIR,
        FIGURE_DIR,
        REPORT_DIR,
    )
    loader=DataLoader(RAW_DATA_PATH)
    df=loader.load_data()
    if df is None:
        return
    
    preprocessor=DataPreprocessor(df)
    preprocessor.full_report()
    eda=EDA(df, FIGURE_DIR)
    eda.run_all()

    feature_engineering=FeatureEngineering(
        df,TARGET_COLUMN
    )
    processed_df=feature_engineering.process()
    feature_engineering.save_processed_dataset(PROCESSED_DATA_PATH)

    X_train, X_test, y_train, y_test=feature_engineering.split_dataset(
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    trainer=ModelTrainer()
    trained_models=trainer.train_models(
        X_train, y_train
    )

    evaluator=Evaluator()
    results=evaluator.evaluate_models(
        trained_models,
        X_test, y_test
    )
    best_model_name=max(
        results, 
        key=lambda model: results[model]["Accuracy"]

    )
    print(f"\nBest Baseline Model:{best_model_name}")
    best_model=trained_models[best_model_name]
    trainer.save_model(
        best_model, MODEL_DIR / "best_model.pkl"
    )
    evaluator.classification_report(
        best_model, 
        X_test,
        y_test
    )
    evaluator.confusion_matrux_plot(
        best_model, 
        X_test,
        y_test,
        FIGURE_DIR / "confusion_matrix.png"
    )
    evaluator.roc_curve_plot(
        best_model,
        X_test,
        y_test,
        FIGURE_DIR / "roc_curve.png"
    )
    evaluator.save_results(
        results, 
        REPORT_DIR / "model_metrices.csv"
    )


# Model performance
    print("\nModel Performance")
    print("="*60)
    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
# hyper parameter

    print("\n"+"="*60)
    print("Hyperparameter Tuning ")
    print("="*60)
    tuner=HyperParameterTuner()
    best_rf=tuner.tune(
        X_train,
        y_train
    )
    trainer.save_model(
        best_rf,
        MODEL_DIR / "Best_random_forest.pkl"
    )
# Corss validations
    cross_validator=CrossValidation()
    cross_validator.evaluate(
        best_rf,
        X_train,
        y_train
    )

# Learning curve
    learning_curve=LeaarningCurve()
    learning_curve.plot(
        best_rf,
        X_train, 
        y_train, 
        FIGURE_DIR / "learning_curve.png"
    )

# Feature importance

    print(type(best_rf))
    print(best_rf)
    feature_importance=FeatureImportance()
    feature_importance.plot(
        best_rf,
        X_train,
        FIGURE_DIR / "feature_importance.png"
    )
# Predictor
    print("\n"+"="*60)
    print("Passenger Prediction")
    print("="*60)
    predictor=Predictor(
        MODEL_DIR /"Best_random_forest.pkl"
    )
    sample_passenger={
        "Pclass":1,
        "Sex":0,
        "Age":28,
        "SibSp":0,
        "Parch":0,
        "Fare":100,
        "Embarked":0
    }
    prediction, probability=predictor.predict(
        sample_passenger
    )
    print("\nPrediction")
    print("Survived" if prediction == 1 else "Did Not Survive")
    if probability is not None:
        print(f"Survival Probability :{probability:.2%}")

    print("\nTraining Shape:", X_train.shape)
    print("Testing Shape :",X_test.shape)



if __name__=="__main__":
    main()