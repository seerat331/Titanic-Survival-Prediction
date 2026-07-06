import pandas as pd 
class DataLoader:
    def __init__(self, file_path):
        self.file_path=file_path
    def load_data(self):
        try:
            df=pd.read_csv(self.file_path)
            print("Dataset Load Sucessfully.")
            return df
        except FileNotFoundError:
            print("Dataset is not found.")
            return None
        except Exception as e:
            print(e)
            return None
        