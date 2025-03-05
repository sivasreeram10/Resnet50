# Resnet50
# banana-disease-detection-Resnet50
# 🍌 ResNet-50-Based Banana Plant Disease Detection  

## Overview  
This project implements a **deep learning model using ResNet-50** to detect diseases in banana plants from leaf images. The model can classify images into different disease categories and help farmers take early preventive measures.  

## Features  
✅ **Deep Learning-Based Disease Detection** – Uses ResNet-50 for high accuracy.  
✅ **Data Augmentation** – Improves model robustness.  
✅ **Pretrained Model (Transfer Learning)** – Faster training and better generalization.  
✅ **Real-Time Predictions** – Can be deployed on edge devices.  
✅ **Scalable & Efficient** – Optimized for mobile and IoT applications.  

## Dataset  
The model is trained on a dataset containing banana leaf images categorized as:  
- **Healthy Leaves**  
- **Diseased Leaves** (Black Sigatoka, Fusarium Wilt, Banana Bunchy Top Virus, etc.)  

## Workflow  
### **1. Data Preprocessing**  
- Resize images to **224x224 pixels** (ResNet-50 input size).  
- Apply **normalization and data augmentation** (rotation, flipping, brightness adjustments).  
- Use **ImageDataGenerator** for efficient data loading.  

### **2. Model Architecture (ResNet-50)**  
The model consists of:  
- **Pretrained ResNet-50 Backbone** (trained on ImageNet).  
- **Custom Fully Connected Layers** for banana disease classification.  
- **Global Average Pooling (GAP) Layer** to reduce parameters.  
- **Softmax Activation** for multi-class classification.  

### **3. Model Training**  
- **Loss Function:** Categorical Cross-Entropy  
- **Optimizer:** Adam (Learning Rate: 0.0001)  
- **Batch Size:** 32  
- **Epochs:** 10  

### **4. Prediction & Deployment**  
- Classifies banana leaf images as **healthy or diseased**.  
- Displays **confidence scores** for each prediction.  
- Can be deployed on **mobile apps, web platforms, and IoT devices**.  

## Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/resnet50-banana-disease-detection.git
   cd resnet50-banana-disease-detection

   Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Train the model:
bash
Copy
Edit
python train.py
Run inference on a sample image:
bash
Copy
Edit
python predict.py --image sample_leaf.jpg
Results
Achieved high accuracy in detecting banana plant diseases.
Low computational cost due to transfer learning.
Future Improvements
🔹 Fine-tune the model with more dataset variations.
🔹 Optimize for edge computing (TensorFlow Lite).
🔹 Implement real-time disease detection on mobile apps.
