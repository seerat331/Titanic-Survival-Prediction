from src.config import (
    RAW_DATA_PATH,
    MODEL_DIR,
    FIGURE_DIR,
    REPORT_DIR
)
from src.utils import create_directories
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.eda import EDA
from src.feature_engineering import FeatureEngineering
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

    X_train, X_test, y_tarin, y_test=feature_engineering.split_dataset(
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    print("\nTraining Shape:", X_train.shape)
    print("Testing Shape :",X_test.shape)

if __name__=="__main__":
    main()