import sys
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from src.logger import logging
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        logging.info("Initializing HuggingFace CLIP Zero-Shot Multimodal Pipeline")
        self.model_id = "openai/clip-vit-base-patch32"
        # We load these lazily or during init. This might take a bit on first run to download.
        self.model = CLIPModel.from_pretrained(self.model_id)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        
        # Hardcoded Comprehensive List of Tamil Nadu Cuisine for Zero-Shot Classification
        self.cuisine_labels = [
            "Chicken Chettinad",
            "Mutton Chukka",
            "South Indian Dosa",
            "Idli and Sambar",
            "Pongal",
            "Madurai Kari Dosai",
            "Malabar Fish Curry",
            "Ambur Biryani",
            "Filter Coffee",
            "Parotta and Salna",
            "Medhu Vada"
        ]

    def predict(self, uploaded_image: Image.Image, user_text_hint: str):
        try:
            logging.info("Executing Genuine Zero-Shot CLIP classification")
            
            # Formulate text prompts to help CLIP
            text_prompts = [f"a photo of authentic {label}" for label in self.cuisine_labels]
            
            # If the user provided a very strong recipe text, we could dynamically append it
            if user_text_hint and len(user_text_hint.strip()) > 3 and user_text_hint != "Mock Text":
                text_prompts.append(f"a photo of {user_text_hint}")
                self.cuisine_labels.append(user_text_hint)

            inputs = self.processor(
                text=text_prompts, 
                images=uploaded_image, 
                return_tensors="pt", 
                padding=True
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract Image-Text similarity logits
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            best_idx = probs.argmax().item()
            predicted_cuisine = self.cuisine_labels[best_idx]
            
            logging.info(f"CLIP Prediction Successful: {predicted_cuisine} with probability {probs[0][best_idx].item()*100:.2f}%")
            return predicted_cuisine
            
        except Exception as e:
            raise CustomException(e, sys)
