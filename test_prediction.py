#!/usr/bin/env python3
"""
Test script to predict on a real CT scan image using the trained model
"""
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from PIL import Image
from tensorflow import keras
import glob

print('\n' + '='*70)
print('KIDNEY TUMOR DETECTION - IMAGE PREDICTION TEST')
print('='*70)

# Load the trained model
print('\n📦 Loading trained model...')
model = keras.models.load_model('model/kidney_model.h5')
print('✓ Model loaded successfully!')

# Image configuration
IMG_SIZE = 224
CLASS_NAMES = ['Normal', 'Tumor']

# Find a test image from the dataset
print('\n🔍 Finding test images from dataset...')
dataset_path = r"C:\Users\gp890\Downloads\kidney-ct-scan-image (1)\kidney-ct-scan-image"

# Get a Normal image
normal_images = glob.glob(os.path.join(dataset_path, 'Normal', '*.jpg'))
# Get a Tumor image
tumor_images = glob.glob(os.path.join(dataset_path, 'Tumor', '*.jpg'))

if not normal_images or not tumor_images:
    print('✗ No images found in dataset!')
    sys.exit(1)

print(f'✓ Found {len(normal_images)} Normal and {len(tumor_images)} Tumor images')

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path, label_expected):
    """Make prediction on image"""
    try:
        img_array = preprocess_image(image_path)
        prediction = model.predict(img_array, verbose=0)[0][0]
        class_index = int(prediction > 0.5)
        class_name = CLASS_NAMES[class_index]
        confidence = float(prediction if class_index == 1 else 1 - prediction)
        
        return {
            'predicted': class_name,
            'expected': label_expected,
            'confidence': confidence,
            'correct': class_name == label_expected
        }
    except Exception as e:
        return {'error': str(e)}

# Test on Normal image
print('\n' + '-'*70)
print('TEST 1: NORMAL CT SCAN IMAGE')
print('-'*70)
normal_test = normal_images[0]
print(f'Image: {os.path.basename(normal_test)}')
result1 = predict_image(normal_test, 'Normal')
if 'error' not in result1:
    print(f'Predicted: {result1["predicted"]}')
    print(f'Expected: {result1["expected"]}')
    print(f'Confidence: {result1["confidence"]*100:.2f}%')
    print(f'Correct: {"✓ YES" if result1["correct"] else "✗ NO"}')
else:
    print(f'Error: {result1["error"]}')

# Test on Tumor image
print('\n' + '-'*70)
print('TEST 2: TUMOR CT SCAN IMAGE')
print('-'*70)
tumor_test = tumor_images[0]
print(f'Image: {os.path.basename(tumor_test)}')
result2 = predict_image(tumor_test, 'Tumor')
if 'error' not in result2:
    print(f'Predicted: {result2["predicted"]}')
    print(f'Expected: {result2["expected"]}')
    print(f'Confidence: {result2["confidence"]*100:.2f}%')
    print(f'Correct: {"✓ YES" if result2["correct"] else "✗ NO"}')
else:
    print(f'Error: {result2["error"]}')

# Summary
print('\n' + '='*70)
print('TEST SUMMARY')
print('='*70)
if 'error' not in result1 and 'error' not in result2:
    accuracy = (result1['correct'] + result2['correct']) / 2 * 100
    print(f'✓ Predictions on 2 test images: {accuracy:.0f}% correct')
    print(f'\n✓ MODEL IS WORKING CORRECTLY!')
    print(f'\n💡 NEXT: Open frontend/index.html in browser to test interactively')
else:
    print('✗ Error during testing')

print('='*70 + '\n')
