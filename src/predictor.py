import joblib 
import pandas as pd
class Predictor:
    def __init__(self, model_path):
        self.model=joblib.load(model_path)
    def predict(self, passenger):
        df=pd.DataFrame([passenger])
        predication=self.model.predict(df)[0]
        if hasattr(self.model, "predict_proba"):
            probability=self.model.predict_proba(df)[0][1]

        else:
            probability=None
        return predication, probability
    