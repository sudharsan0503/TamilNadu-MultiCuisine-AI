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
        
        self.cuisine_details = {
            "Chicken Chettinad": {"region": "Chettinad", "description": "Known for its complex spice mix including star anise, kalpasi, and marathi mokku."},
            "Mutton Chukka": {"region": "Madurai", "description": "Dry pan-roasted tender lamb packed with intense black pepper and curry leaf flavors."},
            "South Indian Dosa": {"region": "Pan-Tamil Nadu", "description": "A crisp, savory crepe made from a fermented batter of ground white gram and rice."},
            "Idli and Sambar": {"region": "Pan-Tamil Nadu", "description": "Steamed rice cakes served with a rich lentil-based vegetable stew."},
            "Pongal": {"region": "Tamil Nadu", "description": "A comforting, ghee-laden blend of boiled rice, yellow lentils, black pepper, and cumin."},
            "Madurai Kari Dosai": {"region": "Madurai", "description": "A thick, fluffy dosa layered with an omelette and spicy minced mutton (kheema)."},
            "Malabar Fish Curry": {"region": "Coastal / Nanjil", "description": "A tangy, coconut-infused fish stew highlighting tamarind and rich coastal spices."},
            "Ambur Biryani": {"region": "Arcot", "description": "Distinctive biryani cooked with jeeraga samba rice and a pungent dried chili paste."},
            "Filter Coffee": {"region": "Tamil Nadu", "description": "Strong, frothy coffee brewed through a traditional metal filter and violently mixed with hot milk."},
            "Parotta and Salna": {"region": "Madurai / Virudhunagar", "description": "Flaky, multi-layered flatbread paired with an intensely savory, thin gravy."},
            "Medhu Vada": {"region": "Tamil Nadu", "description": "Deep-fried, savory lentil donuts perfectly crispy on the outside and fluffy inside."}
        }

    def predict(self, uploaded_image: Image.Image, user_text_hint: str):
        try:
            logging.info("Executing Genuine Zero-Shot CLIP classification")
            
            # Formulate text prompts to help CLIP
            text_prompts = [f"a photo of authentic {label}" for label in self.cuisine_labels]
            
            # Capture dynamic hints
            has_dynamic_hint = False
            if user_text_hint and len(user_text_hint.strip()) > 3 and user_text_hint != "Mock Text":
                text_prompts.append(f"a photo of {user_text_hint}")
                self.cuisine_labels.append(user_text_hint)
                self.cuisine_details[user_text_hint] = {"region": "User Defined", "description": "A special dish identified via user visual-text mapping."}
                has_dynamic_hint = True

            inputs = self.processor(
                text=text_prompts, 
                images=uploaded_image, 
                return_tensors="pt", 
                padding=True
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            best_idx = probs.argmax().item()
            predicted_cuisine = self.cuisine_labels[best_idx]
            details = self.cuisine_details[predicted_cuisine]
            
            # Remove the appended dynamic label to prevent memory leaks across sessions
            if has_dynamic_hint:
                self.cuisine_labels.pop()
                del self.cuisine_details[user_text_hint]
            
            logging.info(f"CLIP Prediction Successful: {predicted_cuisine}")
            
            return {
                "name": predicted_cuisine,
                "probability": round(probs[0][best_idx].item() * 100, 2),
                "region": details["region"],
                "description": details["description"]
            }
            
        except Exception as e:
            raise CustomException(e, sys)
