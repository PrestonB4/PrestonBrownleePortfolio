from flask import Flask, jsonify
from flask_cors import CORS
import sys
import os

# Add apps directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'apps'))

app = Flask(__name__)
CORS(app)

# Import and register blueprints
from apps.nlp_sentiment.nlp_blueprint import nlp_bp
from apps.nba_team.nba_team_blueprint import nba_bp
from apps.image_classifier.image_classifier_blueprint import image_bp

app.register_blueprint(nlp_bp)
app.register_blueprint(nba_bp)
app.register_blueprint(image_bp)

@app.route('/')
def index():
    return jsonify({
        "message": "Portfolio API Hub",
        "apps": [
            {"name": "NLP Sentiment Analyzer", "endpoint": "/api/nlp"},
            {"name": "NBA Team Optimizer", "endpoint": "/api/nba"},
            {"name": "Image Classifier (Cats vs Dogs)", "endpoint": "/api/image-classifier"}
        ]
    })

@app.route('/api/health')
def health():
    return jsonify({"status": "ok", "message": "Portfolio API is running"})

if __name__ == "__main__":
    app.run(debug=True)
