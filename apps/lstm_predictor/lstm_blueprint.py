"""
Flask Blueprint for LSTM Next Word Prediction
"""

from flask import Blueprint, request, jsonify
import numpy as np
import pickle
import string
import os
import gc
import threading
import time
import psutil

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras

lstm_bp = Blueprint('lstm', __name__, url_prefix='/api/lstm')

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

# Global variables for lazy loading
_model = None
_tokenizer = None
_config = None
_last_used = None
_cleanup_timer = None
_lock = threading.Lock()

# Cleanup timeout in seconds (10 minutes of inactivity)
CLEANUP_TIMEOUT = 600

def cleanup_model():
    """Unload model from memory after timeout"""
    global _model, _tokenizer, _config, _cleanup_timer
    with _lock:
        if _model is not None and time.time() - _last_used > CLEANUP_TIMEOUT:
            print("LSTM model inactive for 10 minutes, cleaning up...")
            _model = None
            _tokenizer = None
            _config = None
            gc.collect()
            print("LSTM model unloaded from memory")
        _cleanup_timer = None

def schedule_cleanup():
    """Schedule model cleanup after timeout"""
    global _cleanup_timer
    if _cleanup_timer:
        _cleanup_timer.cancel()
    _cleanup_timer = threading.Timer(CLEANUP_TIMEOUT, cleanup_model)
    _cleanup_timer.daemon = True
    _cleanup_timer.start()

def get_model_artifacts():
    """Lazy load the trained model, tokenizer, and config."""
    global _model, _tokenizer, _config, _last_used

    with _lock:
        if _model is None:
            # Measure memory before loading
            mem_before = get_memory_usage()
            print(f"Memory before loading LSTM: {mem_before:.2f} MB")

            # Path to model artifacts
            base_path = os.path.dirname(os.path.abspath(__file__))
            models_path = os.path.join(base_path, 'models')

            print("Loading LSTM model artifacts...")

            # Load model
            model_path = os.path.join(models_path, 'best_lstm_model.keras')
            _model = keras.models.load_model(model_path)
            print(f"✓ Loaded LSTM model from {model_path}")

            # Load tokenizer
            tokenizer_path = os.path.join(models_path, 'tokenizer.pickle')
            with open(tokenizer_path, 'rb') as f:
                _tokenizer = pickle.load(f)
            print(f"✓ Loaded tokenizer from {tokenizer_path}")

            # Load config
            config_path = os.path.join(models_path, 'model_config.pickle')
            with open(config_path, 'rb') as f:
                _config = pickle.load(f)
            print(f"✓ Loaded config from {config_path}")

            # Measure memory after loading
            mem_after = get_memory_usage()
            mem_used = mem_after - mem_before
            print(f"Memory after loading LSTM: {mem_after:.2f} MB")
            print(f"LSTM model uses: {mem_used:.2f} MB")
            print("LSTM artifacts loaded successfully!")

        _last_used = time.time()
        schedule_cleanup()
        return _model, _tokenizer, _config

def predict_next_word(text, num_predictions=5):
    """
    Predict the next word given input text.

    Args:
        text: Input text string
        num_predictions: Number of top predictions to return

    Returns:
        list: Top predictions with words and probabilities
    """
    model, tokenizer, config = get_model_artifacts()
    sequence_length = config['sequence_length']

    # Preprocess input text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize
    tokens = tokenizer.texts_to_sequences([text])[0]

    # Pad or truncate to sequence_length
    if len(tokens) < sequence_length:
        tokens = [0] * (sequence_length - len(tokens)) + tokens
    else:
        tokens = tokens[-sequence_length:]

    # Reshape for model
    tokens = np.array(tokens).reshape(1, sequence_length)

    # Predict
    predictions = model.predict(tokens, verbose=0)[0]

    # Get top predictions, filtering out OOV (index 1)
    top_indices = predictions.argsort()[::-1]
    results = []

    for idx in top_indices:
        if idx == 1:  # Skip OOV token
            continue

        word = None
        for w, i in tokenizer.word_index.items():
            if i == idx:
                word = w
                break

        if word:
            results.append({
                'word': word,
                'probability': float(predictions[idx])
            })

        if len(results) >= num_predictions:
            break

    return results

def complete_sentence(seed_text, num_words=5):
    """
    Complete a sentence by predicting multiple words.

    Args:
        seed_text: Starting text
        num_words: Number of words to generate

    Returns:
        str: Completed sentence
    """
    result = seed_text

    for _ in range(num_words):
        predictions = predict_next_word(result, num_predictions=1)
        if predictions:
            next_word = predictions[0]['word']
            result = result + " " + next_word
        else:
            break

    return result

# API Routes

@lstm_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': _model is not None
    })

@lstm_bp.route('/memory', methods=['GET'])
def memory_stats():
    """Get current memory usage statistics."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return jsonify({
        'total_memory_mb': mem_info.rss / 1024 / 1024,
        'model_loaded': _model is not None,
        'message': 'Load the model by calling /predict or /complete to see memory increase'
    })

@lstm_bp.route('/predict', methods=['POST'])
def predict():
    """
    Predict next word(s) for given text.

    Request body:
        {
            "text": "I want to",
            "num_predictions": 5
        }

    Response:
        {
            "input": "I want to",
            "predictions": [
                {"word": "be", "probability": 0.1135},
                ...
            ]
        }
    """
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400

        text = data['text']
        num_predictions = data.get('num_predictions', 5)

        predictions = predict_next_word(text, num_predictions)

        return jsonify({
            'input': text,
            'predictions': predictions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@lstm_bp.route('/complete', methods=['POST'])
def complete():
    """
    Complete a sentence with multiple predicted words.

    Request body:
        {
            "text": "I want to",
            "num_words": 5
        }

    Response:
        {
            "input": "I want to",
            "completed": "I want to be a lot of people",
            "num_words_added": 5
        }
    """
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400

        text = data['text']
        num_words = data.get('num_words', 5)

        completed = complete_sentence(text, num_words)

        return jsonify({
            'input': text,
            'completed': completed,
            'num_words_added': num_words
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@lstm_bp.route('/config', methods=['GET'])
def get_config():
    """Get model configuration."""
    model, tokenizer, config = get_model_artifacts()
    return jsonify({
        'sequence_length': config['sequence_length'],
        'vocab_size': config['vocab_size'],
        'embedding_dim': config['embedding_dim']
    })
