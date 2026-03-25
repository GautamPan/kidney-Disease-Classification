"""
Kidney Tumor Detection - Flask Backend API & Web Server
Full-stack web application for kidney tumor detection
- Serves interactive web interface at http://localhost:5000
- Provides REST API for predictions
Model: Trained CNN (kidney_model.h5)
"""

import os
import numpy as np
from PIL import Image
from io import BytesIO
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras

# ==================== CONFIGURATION ====================
MODEL_PATH = r"model/kidney_model.h5"
IMG_SIZE = 224
CLASS_NAMES = ['Normal', 'Tumor']

# Get the parent directory (project root)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
STATIC_DIR = os.path.join(PROJECT_ROOT, 'frontend')

# ==================== FLASK APP SETUP ====================
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')
CORS(app)  # Enable CORS for cross-origin requests

# Global variable to store loaded model
model = None

# ==================== MODEL LOADING ====================
def load_model():
    """
    Load the trained kidney tumor detection model on startup
    """
    global model
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from: {MODEL_PATH}")
            model = keras.models.load_model(MODEL_PATH)
            print("✓ Model loaded successfully!")
            return True
        else:
            print(f"✗ ERROR: Model not found at {MODEL_PATH}")
            print(f"Please run training script first: python model/train.py")
            return False
    except Exception as e:
        print(f"✗ ERROR loading model: {str(e)}")
        traceback.print_exc()
        return False

# ==================== IMAGE PREPROCESSING ====================
def preprocess_image(image_file):
    """
    Preprocess image: convert to RGB, resize to 224x224, normalize to [0,1]
    
    Args:
        image_file: Image file from request
    
    Returns:
        Preprocessed image array ready for model
    """
    try:
        # Open image
        img = Image.open(image_file).convert('RGB')
        
        # Resize to model input size
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy array
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Serve the frontend HTML"""
    frontend_file = os.path.join(STATIC_DIR, 'index.html')
    try:
        with open(frontend_file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return jsonify({'error': f'Frontend not found at {frontend_file}'}), 404

# ==================== PREDICTION ====================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if kidney CT scan shows tumor or normal
    
    Request:
        - Method: POST
        - Content-Type: multipart/form-data
        - Key: 'file' (image file)
    
    Response:
        JSON with:
        - prediction: "Normal" or "Tumor"
        - confidence: float (0-1)
    """
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please restart the application.'
        }), 500
    
    # Validate file upload
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided. Please upload an image.'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
    
    # Validate file type
    allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
    if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return jsonify({
            'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'
        }), 400
    
    try:
        # Preprocess image
        img_array = preprocess_image(file)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # Interpret prediction (binary: 0=Normal, 1=Tumor)
        class_index = int(prediction > 0.5)
        class_name = CLASS_NAMES[class_index]
        confidence = float(prediction if class_index == 1 else 1 - prediction)
        
        return jsonify({
            'prediction': class_name,
            'confidence': round(confidence, 4),
            'confidence_percent': round(confidence * 100, 2)
        }), 200
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"ERROR in prediction: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

# ==================== HEALTH CHECK ====================
@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint to verify API is running and model is loaded
    """
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None
    }), 200

# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ==================== APP STARTUP ====================
@app.before_request
def startup_check():
    """Ensure model is loaded before processing requests"""
    global model
    if model is None:
        # Try to load model on first request
        load_model()

if __name__ == '__main__':
    print("\n" + "="*70)
    print("KIDNEY TUMOR DETECTION - WEB APPLICATION")
    print("="*70)
    
    # Load model
    if load_model():
        print("\n✓ Server starting...")
        print("\n" + "="*70)
        print("🌐 VISIT: http://localhost:5000")
        print("="*70)
        print("\nEndpoints:")
        print("  • GET  http://localhost:5000/           → Web Interface")
        print("  • POST http://localhost:5000/predict    → API Prediction")
        print("  • GET  http://localhost:5000/health     → Health Check")
        print("\n" + "="*70 + "\n")
        
        # Run Flask app
        app.run(debug=False, host='localhost', port=5000, use_reloader=False)
    else:
        print("\n✗ ERROR: Cannot start server without trained model.")
        print("Please run: python model/train.py")
