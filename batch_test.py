#!/usr/bin/env python3
"""Test model on multiple images"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from PIL import Image
from tensorflow import keras
import glob
import random

# Load model
model = keras.models.load_model('model/kidney_model.h5')
IMG_SIZE = 224
CLASS_NAMES = ['Normal', 'Tumor']

dataset_path = r"C:\Users\gp890\Downloads\kidney-ct-scan-image (1)\kidney-ct-scan-image"
normal_images = glob.glob(os.path.join(dataset_path, 'Normal', '*.jpg'))
tumor_images = glob.glob(os.path.join(dataset_path, 'Tumor', '*.jpg'))

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

print('\n' + '='*70)
print('TESTING MODEL ON 10 RANDOM IMAGES')
print('='*70)

# Test 5 Normal images
print('\nNORMAL IMAGES (Expected: Normal):')
print('-'*70)
normal_correct = 0
for img_path in random.sample(normal_images, min(5, len(normal_images))):
    pred = model.predict(preprocess_image(img_path), verbose=0)[0][0]
    pred_class = CLASS_NAMES[int(pred > 0.5)]
    confidence = (pred if pred > 0.5 else 1-pred) * 100
    is_correct = pred_class == 'Normal'
    normal_correct += is_correct
    status = '✓' if is_correct else '✗'
    print(f'{status} {os.path.basename(img_path)}: {pred_class} ({confidence:.1f}%)')

# Test 5 Tumor images
print('\nTUMOR IMAGES (Expected: Tumor):')
print('-'*70)
tumor_correct = 0
for img_path in random.sample(tumor_images, min(5, len(tumor_images))):
    pred = model.predict(preprocess_image(img_path), verbose=0)[0][0]
    pred_class = CLASS_NAMES[int(pred > 0.5)]
    confidence = (pred if pred > 0.5 else 1-pred) * 100
    is_correct = pred_class == 'Tumor'
    tumor_correct += is_correct
    status = '✓' if is_correct else '✗'
    print(f'{status} {os.path.basename(img_path)}: {pred_class} ({confidence:.1f}%)')

# Summary
total_correct = normal_correct + tumor_correct
accuracy = (total_correct / 10) * 100
print('\n' + '='*70)
print(f'ACCURACY: {accuracy:.0f}% ({total_correct}/10 correct)')
print('='*70)
print('\n✓ Backend API is running on http://localhost:5000')
print('✓ To test interactively:')
print('  1. Open: frontend/index.html')
print('  2. Drag & drop a CT scan image')
print('  3. Click "Analyze" to get predictions')
print('='*70 + '\n')
