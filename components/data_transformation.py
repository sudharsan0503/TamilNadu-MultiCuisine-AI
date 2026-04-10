import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

# Mock Dataset Class to simulate Multi-Modal Image/Text Loading
class TNCuisineDataset(Dataset):
    def __init__(self, data_frame):
        self.data = data_frame
        # Mocking labels mapping
        self.label_map = {label: i for i, label in enumerate(self.data['dish_name_en'].unique())}
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Vision Feature (Mocking a 3-channel 224x224 ViT input image tensor)
        image_tensor = torch.randn(3, 224, 224) 
        
        # 2. Textual Feature (Mocking a 16-token text sequence)
        text_tensor = torch.randint(0, 1000, (16,))
        
        # 3. Label
        label_text = self.data.iloc[idx]['dish_name_en']
        label = self.label_map[label_text]
        
        return image_tensor, text_tensor, torch.tensor(label, dtype=torch.long)

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Creating Custom PyTorch Datasets")
            train_dataset = TNCuisineDataset(train_df)
            test_dataset = TNCuisineDataset(test_df)
            
            logging.info("Creating PyTorch DataLoaders")
            train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

            # In a real model, we save the tokenizer/label mapping here
            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj={'label_map': train_dataset.label_map}
            )

            return train_dataloader, test_dataloader, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
