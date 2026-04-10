import sys
import torch
from src.logger import logging
from src.exception import CustomException
from components.model_trainer import TamilCuisineMultimodal

class ModelEvaluation:
    def __init__(self):
        pass

    def initiate_model_evaluation(self, test_loader, model_path):
        try:
            logging.info("Initiating Final Model Evaluation on Checkpoint")
            
            # Load Architecture
            model = TamilCuisineMultimodal(num_classes=5)
            # Load Weights
            model.load_state_dict(torch.load(model_path))
            model.eval()

            correct = 0
            total = 0

            with torch.no_grad():
                for images, texts, labels in test_loader:
                    outputs = model(images, texts)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total if total > 0 else 0
            logging.info(f"Model Evaluation Complete. Test Accuracy on unseen mock data: {accuracy:.2f}%")
            
            return accuracy
            
        except Exception as e:
            raise CustomException(e, sys)
