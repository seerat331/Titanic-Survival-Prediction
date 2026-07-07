from src.exception import DatasetNotFoundError
from src.logger import logger
import pandas as pd 
class DataLoader:
    def __init__(self, file_path):
        self.file_path=file_path
    def load_data(self):
        try:
            df=pd.read_csv(self.file_path)
            logger.info("Dataset loaded successfully.")
            return df
        except FileNotFoundError:
            logger.error("Dataset not found.")
            raise DatasetNotFoundError(
                f"Dataset not found at {self.file_path}"
            )
        except Exception as e:
            logger.exception(
                f"Unexpected error while loading dataset: {e}"
                )
            return None
        