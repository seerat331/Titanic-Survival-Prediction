from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent.parent
RAW_DATA_PATH=BASE_DIR / "data" / "raw" / "Titanic-Dataset.csv"
PREDICTION_DIR=BASE_DIR / "outputs" /"predictions"
PROCESSED_DATA_PATH=(
    BASE_DIR
    / "data"
    /"processed"
    /"processedtitanic.csv"
)
MODEL_DIR=BASE_DIR / "models"
FIGURE_DIR=BASE_DIR / "outputs" / "figures"
REPORT_DIR=BASE_DIR / "outputs" / "reports"
 

TARGET_COLUMN="Survived"
TEST_SIZE=0.2
RANDOM_STATE=42
SHOW_FULL_REPORT = True
SHOW_EDA = True