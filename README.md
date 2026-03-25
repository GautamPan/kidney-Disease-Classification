# Kidney Tumor Detection - Web Application

A complete end-to-end web application for detecting kidney tumors from CT scan images using a Deep Convolutional Neural Network (CNN).

## 📋 Project Structure

```
Minor_p/
├── model/
│   ├── train.py                 # CNN training script
│   ├── kidney_model.h5          # Trained model (generated after training)
│   ├── training_curves.png      # Training history plots (generated)
│   └── confusion_matrix.png     # Confusion matrix (generated)
├── backend/
│   └── app.py                   # Flask API server
├── frontend/
│   └── index.html               # Single-page web application
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 2. **Prepare Dataset**

Ensure your dataset is organized as:
```
C:\Users\gp890\Downloads\kidney-ct-scan-image (1)\
├── Normal/          # Normal CT scan images
└── Tumor/           # CT scan images with tumors
```

### 3. **Train the Model**

```bash
python model/train.py
```

This will:
- Load and augment images from the dataset
- Train the CNN model (3 Conv2D blocks)
- Save the trained model as `model/kidney_model.h5`
- Generate training curves and confusion matrix
- Print classification report and metrics

**Expected output:**
- `model/kidney_model.h5` - Trained model
- `model/training_curves.png` - Accuracy/Loss plots
- `model/confusion_matrix.png` - Validation metrics

### 4. **Start the Backend API**

```bash
python backend/app.py
```

The Flask server will start on `http://localhost:5000`

Expected output:
```
KIDNEY TUMOR DETECTION - FLASK BACKEND
============================================================
Loading model from: model/kidney_model.h5
Model loaded successfully!

Starting Flask server on http://localhost:5000
Endpoint: POST http://localhost:5000/predict
============================================================
```

### 5. **Open Frontend**

Simply open `frontend/index.html` in your web browser:
- Option 1: Double-click the file
- Option 2: Right-click → Open with → Web Browser
- Option 3: Use your file explorer to navigate to the file

## 🎯 How to Use

1. **Upload Image**: Drag & drop a CT scan image or click to browse
2. **Preview**: View the selected image before analysis
3. **Analyze**: Click the "Analyze" button
4. **Results**: See prediction (Normal/Tumor) with confidence percentage
5. **Clear**: Start over with a new image

## 🧠 Model Architecture

The CNN consists of:

```
Input (224x224x3)
    ↓
Conv2D(32) + ReLU + MaxPooling + BatchNorm (112x112x32)
    ↓
Conv2D(64) + ReLU + MaxPooling + BatchNorm (56x56x64)
    ↓
Conv2D(128) + ReLU + MaxPooling + BatchNorm (28x28x128)
    ↓
Flatten (100352)
    ↓
Dense(256) + ReLU + Dropout(0.5)
    ↓
Dense(1) + Sigmoid
    ↓
Output: Binary Classification (Normal=0, Tumor=1)
```

## 📊 Training Details

- **Image Size**: 224x224 pixels (RGB)
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Train/Val Split**: 80/20
- **Augmentation**: Horizontal flip, zoom, rotation, shear, shifts
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Callbacks**: EarlyStopping, ModelCheckpoint

## 🔌 API Endpoints

### POST /predict
Predict kidney tumor from CT scan image

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response (Success - 200):**
```json
{
  "prediction": "Normal" or "Tumor",
  "confidence": 0.9234,
  "confidence_percent": 92.34
}
```

**Response (Error - 400/500):**
```json
{
  "error": "Error description"
}
```

### GET /health
Check if API is running and model is loaded

**Response:**
```json
{
  "status": "running",
  "model_loaded": true
}
```

## ✨ Features

✅ **Model Training**
- CNN with 3 convolutional blocks
- Data augmentation for better generalization
- Early stopping to prevent overfitting
- Automatic model checkpointing

✅ **Backend API**
- Flask REST API with CORS support
- Image validation and preprocessing
- Error handling and logging
- Health check endpoint

✅ **Frontend UI**
- Clean, dark-themed interface
- Drag-and-drop image upload
- Image preview before analysis
- Loading spinner during prediction
- Color-coded results (Green=Normal, Red=Tumor)
- Confidence visualization with progress bar
- Responsive design for mobile/tablet
- No dependencies - Pure HTML/CSS/JavaScript

## 🛠️ Troubleshooting

### Model not found error
```
ERROR: Model not found at model/kidney_model.h5
Run: python model/train.py
```

### Connection refused error
Make sure backend is running:
```bash
python backend/app.py
```

### CORS error in browser
The backend has CORS enabled, but ensure:
- Frontend is opened via file:// or http://
- Backend is running on http://localhost:5000

### Out of memory during training
Reduce `BATCH_SIZE` in `model/train.py` (default: 32)

### Poor prediction accuracy
- Ensure dataset has sufficient images (recommend 100+ per class)
- Check image quality and resolution
- Consider training for more epochs (increase `EPOCHS`)
- Verify data is properly split into Normal/Tumor folders

## 📦 Dependencies

- **tensorflow** (2.13.0) - Deep learning framework
- **flask** (3.0.0) - Web framework
- **flask-cors** (4.0.0) - CORS support
- **numpy** (1.24.3) - Numerical computing
- **pillow** (10.0.0) - Image processing
- **scikit-learn** (1.3.0) - ML metrics
- **matplotlib** (3.7.2) - Plotting

## 📝 Notes

- The application uses binary classification (Normal vs Tumor)
- Images are preprocessed to 224x224 RGB
- Predictions are normalized to confidence between 0-1
- Model uses sigmoid activation for binary classification
- Training includes data augmentation to improve robustness
- Early stopping prevents overfitting based on validation loss

## 🎓 Educational Purpose

This application is designed for educational purposes to demonstrate:
- CNN architecture design and implementation
- Medical image classification
- Full-stack web application development
- Model training and evaluation
- REST API development
- Frontend-backend integration

## ⚠️ Disclaimer

This application is for **demonstration and educational purposes only**. It should not be used for actual medical diagnosis. Always consult with qualified medical professionals for medical decisions.

## 📞 Support

For issues or questions, check:
1. Dataset path is correct
2. All dependencies are installed
3. Backend server is running
4. Image format is supported (JPG, PNG, GIF, BMP)
5. No errors in console logs (browser Dev Tools: F12)

---

**Happy Analyzing! 🧬🔬**
