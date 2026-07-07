from src.exception import PredictionError
from src.logger import logger
import joblib
import pandas as pd


class Predictor:
    def __init__(self, model_path):
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model LOaded successfully from {model_path}")
        except Exception as e:
            logger.exception("Unable to load model.")
            raise PredictionError(f"Unable to load model: {e}")

    def predict(self, passenger_data):
        try:
            prediction = self.model.predict(passenger_data)[0]
            probability = self.model.predict_proba(passenger_data)[0][1]
            logger.info("Prediction completed successfully.")
            return prediction, probability
        except Exception as e:
            logger.exception("Prediction Failed.")
            raise PredictionError(f"Prediction failed: {e}")

    def display_prediction(self, prediction, probability):
        print("\n" + "=" * 60)
        print("PASSENGER PREDICTION")
        print("=" * 60)

        if prediction == 1:
            print("Prediction: Survived")
        else:
            print("Prediction :Did Not Survive")
        print(f"Survival Probability: {probability:.2%}")

    def save_prediction(self, prediction, probability, save_path):
        prediction_df = pd.DataFrame(
            {
                "Prediction": ["Survived" if prediction == 1 else "Did Not Survive"],
                "Probability": [round(probability, 4)],
            }
        )
        prediction_df.to_csv(save_path, index=False)

        logger.info(f"Predication saved to {save_path}")
        print(f"\nPredicationsaved to:\n{save_path}")
