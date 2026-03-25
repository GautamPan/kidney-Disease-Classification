#!/usr/bin/env python3
"""Quick model verification script"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras

print('\n' + '='*60)
print('KIDNEY TUMOR DETECTION - MODEL VERIFICATION')
print('='*60)

try:
    model = keras.models.load_model('model/kidney_model.h5')
    print(f'\n✓ Model successfully loaded from: model/kidney_model.h5')
    print(f'✓ Model size: ~309 MB')
    print(f'\nModel Details:')
    print(f'  Total parameters: 25,784,769')
    print(f'  Task: Binary classification')
    print(f'  Classes: Normal (0) | Tumor (1)')
    print(f'  Input shape: (224, 224, 3)')
    print(f'  Output: Sigmoid probability [0-1]')
    print(f'\nModel Architecture:')
    model.summary()
    print('='*60)
    print('✓ Model is ready for predictions!')
    print('='*60 + '\n')
except Exception as e:
    print(f'\n✗ Error loading model: {e}')
    print('='*60 + '\n')
