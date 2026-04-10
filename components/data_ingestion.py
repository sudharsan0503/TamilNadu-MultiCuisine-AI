import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Mocking the TN Multi-Cuisine Dataset for Phase 1
            mock_data = {
                'dish_id': ['TN_CH_001', 'TN_KO_002', 'TN_TJ_003', 'TN_MD_004', 'TN_NJ_005'],
                'dish_name_en': ['Chicken Chettinad', 'Arisiyum Paruppum Sadam', 'Ashoka Halwa', 'Kari Dosai', 'Ulunthanchoru'],
                'region': ['Chettinad', 'Kongunadu', 'Thanjavur', 'Madurai', 'Nanjil Nadu'],
                'category': ['Non-Veg Gravy', 'Main Meals', 'Sweets', 'Street Food', 'Main Meals']
            }
            df = pd.DataFrame(mock_data)
            logging.info("Created mock dataset as pandas dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")
            
            # Use small test size because dataset is very small
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
