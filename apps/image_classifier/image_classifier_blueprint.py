from flask import Blueprint, request, jsonify, send_file
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import io
from PIL import Image
import random
from pathlib import Path
import base64
import gc
import threading
import time

# Create blueprint
image_bp = Blueprint('image_classifier', __name__, url_prefix='/api/image-classifier')

# Lazy loading - model loaded only when first used
_model = None
_class_names = None
_last_used = None
_cleanup_timer = None
_lock = threading.Lock()

# Cleanup timeout in seconds (10 minutes of inactivity)
CLEANUP_TIMEOUT = 600

def cleanup_model():
    """Unload model from memory after timeout"""
    global _model, _class_names, _cleanup_timer
    with _lock:
        if _model is not None and time.time() - _last_used > CLEANUP_TIMEOUT:
            print("Image classifier model inactive for 10 minutes, cleaning up...")
            _model = None
            _class_names = None
            gc.collect()
            print("Image classifier model unloaded from memory")
        _cleanup_timer = None

def schedule_cleanup():
    """Schedule model cleanup after timeout"""
    global _cleanup_timer
    if _cleanup_timer:
        _cleanup_timer.cancel()
    _cleanup_timer = threading.Timer(CLEANUP_TIMEOUT, cleanup_model)
    _cleanup_timer.daemon = True
    _cleanup_timer.start()

def get_model():
    """Lazy load the CNN model to save memory"""
    global _model, _class_names, _last_used
    with _lock:
        if _model is None:
            print("Loading Image Classifier model...")
            # Load the trained model
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'improved_model_catsdogs.keras')
            _model = keras.models.load_model(model_path)

            # Load class names
            class_names_path = os.path.join(os.path.dirname(__file__), 'models', 'class_names.txt')
            with open(class_names_path, 'r') as f:
                _class_names = [line.strip() for line in f.readlines()]
            print("Image Classifier model loaded successfully")

        _last_used = time.time()
        schedule_cleanup()
        return _model, _class_names

def preprocess_image(image_bytes):
    """Preprocess image for model prediction"""
    # Load image from bytes
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize to model input size
    img = img.resize((224, 224))

    # Convert to array and normalize
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    return img_array

@image_bp.route('/predict', methods=['POST'])
def predict():
    """Predict if image is a cat or dog"""
    # Check if file was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read image bytes
        image_bytes = file.read()

        # Preprocess image
        img_array = preprocess_image(image_bytes)

        # Load model and make prediction
        model, class_names = get_model()
        predictions = model.predict(img_array, verbose=0)

        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]

        # Get probabilities for both classes
        cat_prob = float(predictions[0][0])
        dog_prob = float(predictions[0][1])

        return jsonify({
            "prediction": predicted_class.capitalize(),
            "confidence": confidence,
            "probabilities": {
                "cats": cat_prob,
                "dogs": dog_prob
            }
        })

    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

@image_bp.route('/random', methods=['GET'])
def get_random_image():
    """Get a random image from the dataset"""
    try:
        # Path to dataset
        dataset_path = Path(__file__).parent.parent.parent / 'NeuralNetwork' / 'data_large' / 'test'

        # Check if dataset exists
        if not dataset_path.exists():
            return jsonify({"error": "Dataset not found"}), 404

        # Get random class (cats or dogs)
        class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        if not class_dirs:
            return jsonify({"error": "No images found in dataset"}), 404

        random_class_dir = random.choice(class_dirs)

        # Get random image from that class
        images = list(random_class_dir.glob('*.jpg'))
        if not images:
            return jsonify({"error": "No .jpg images found"}), 404

        random_image_path = random.choice(images)

        # Read and encode image as base64
        with open(random_image_path, 'rb') as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        return jsonify({
            "image": f"data:image/jpeg;base64,{image_base64}",
            "actual_class": random_class_dir.name.capitalize()
        })

    except Exception as e:
        return jsonify({"error": f"Error getting random image: {str(e)}"}), 500

@image_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "app": "Image Classifier (Cats vs Dogs)"})
