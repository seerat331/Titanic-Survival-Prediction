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

def main():
    create_directories(
        MODEL_DIR,
        FIGURE_DIR,
        REPORT_DIR
    )
    loader=DataLoader(RAW_DATA_PATH)
    df=loader.load_data()
    if df is None:
        return
    
    preprocessor=DataPreprocessor(df)
    preprocessor.full_report()
    eda=EDA(df, FIGURE_DIR)
    eda.run_all()

if __name__=="__main__":
    main()