import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "artifacts/.gitkeep",
    "components/__init__.py",
    "components/data_ingestion.py",
    "components/data_transformation.py",
    "components/model_trainer.py",
    "components/model_evaluation.py",
    "pipeline/__init__.py",
    "pipeline/train_pipeline.py",
    "pipeline/predict_pipeline.py",
    "notebooks/.gitkeep",
    "research/.gitkeep",
    "src/__init__.py",
    "src/logger.py",
    "src/exception.py",
    "src/utils.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} is already exists")
