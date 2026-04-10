from flask import Flask, request, render_template
from pipeline.predict_pipeline import PredictPipeline
import sys

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

from flask import jsonify
from PIL import Image
import io

# Initialize Global Predictor to avoid reloading 600MB model on every POST
try:
    global_predictor = PredictPipeline()
except Exception as e:
    global_predictor = None

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    try:
        recipe_text = request.form.get('recipe_text', 'Mock Text')
        image_file = request.files.get('dish_image')
        
        if not image_file or image_file.filename == '':
            return jsonify({'success': False, 'error': "You must upload an actual image for the model to understand."})

        # Load physical image bytes into PIL memory
        img_bytes = image_file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # We assume form data would contain raw image/text inputs. 
        if global_predictor is None:
            # Fallback initialization if it failed on startup
            predictor = PredictPipeline()
        else:
            predictor = global_predictor
            
        final_pred = predictor.predict(pil_img, recipe_text)

        return jsonify({'success': True, 'prediction': final_pred, 'text': recipe_text})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
