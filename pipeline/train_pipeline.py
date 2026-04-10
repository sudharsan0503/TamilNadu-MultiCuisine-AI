import sys
from src.logger import logging
from src.exception import CustomException
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer
from components.model_evaluation import ModelEvaluation

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        logging.info("================== PIPELINE TRIGGERED ==================")
        try:
            # Phase 1: Data Ingestion
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            # Phase 2: Data Transformation (Multimodal PyTorch Loaders)
            data_transformation = DataTransformation()
            train_loader, test_loader, preprocessor_path = data_transformation.initiate_data_transformation(train_path, test_path)

            # Phase 2: Model Training
            model_trainer = ModelTrainer()
            model_path = model_trainer.initiate_model_trainer(train_loader, test_loader)

            # Phase 3: Model Evaluation
            model_eval = ModelEvaluation()
            accuracy = model_eval.initiate_model_evaluation(test_loader, model_path)

            logging.info("================== PIPELINE COMPLETED MULTI-MODAL RUN SUCCESSFULLY ==================")
            return accuracy
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
