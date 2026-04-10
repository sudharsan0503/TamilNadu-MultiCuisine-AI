import os
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pth")

# Multimodal Fusion Mock Architecture
class TamilCuisineMultimodal(nn.Module):
    def __init__(self, num_classes=5):
        super(TamilCuisineMultimodal, self).__init__()
        # 1. Vision Backbone Mock (Outputs 512-dim embedding)
        self.vision_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # 2. Text Backbone Mock (Outputs 128-dim embedding via Embedding bag)
        self.text_embedding = nn.Embedding(1000, 16)
        self.text_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 16, 128),
            nn.ReLU()
        )
        
        # 3. Cross-Attention Fusion Mock Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, img, text):
        vision_feat = self.vision_extractor(img)
        text_feat = self.text_extractor(self.text_embedding(text))
        
        # Concatenate features
        fused = torch.cat((vision_feat, text_feat), dim=1)
        output = self.fusion_layer(fused)
        return output

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_loader, test_loader):
        try:
            logging.info("Initializing Final Multimodal Model Architectures")
            # We assume 5 classes based on our dummy dataset
            model = TamilCuisineMultimodal(num_classes=5)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            epochs = 1
            logging.info(f"Starting Training Loop for {epochs} epoch(s)")

            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                for batch_idx, (images, texts, labels) in enumerate(train_loader):
                    optimizer.zero_grad()
                    outputs = model(images, texts)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                
                avg_loss = running_loss / len(train_loader)
                logging.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

            logging.info("Training complete. Saving Final Model weight artifacts (model.pth)")
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            torch.save(model.state_dict(), self.model_trainer_config.trained_model_file_path)

            return self.model_trainer_config.trained_model_file_path
            
        except Exception as e:
            raise CustomException(e, sys)
