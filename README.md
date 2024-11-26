# Comparative Analysis of The Deepfake Classification Performance on Deepfake and Real Images Dataset Based on Human Faces Using Xception Network,Vision Transformers (ViT), Efficient Network, Mobile Network, and Convolutional Neural Network(CNN) Deep Learning Models

This repository focuses on the **comparative analysis of deepfake classification performance** using deep learning models on a dataset containing **deepfake and real images of human faces**. The models used include:

- **Xception Network**
- **Vision Transformers (ViT)**
- **Efficient Network**
- **Mobile Network**
- **Convolutional Neural Network (CNN)**

## 📜 Abstract

Deepfake technology poses significant challenges in detecting manipulated media, especially with the advancements in generative models. This project investigates the performance of various state-of-the-art deep learning models for deepfake classification on a dataset of real and manipulated face images. The analysis evaluates their accuracy, precision, recall, F1-score, and computational efficiency.

## 📁 Dataset

The dataset consists of **deepfake images and real images of human faces**. Key attributes include:

- **Real Images**: High-quality, unaltered human face images.
- **Deepfake Images**: Face-swapped or AI-generated manipulations using state-of-the-art generative models.

### Dataset Highlights:
- Balanced classes for real and deepfake images.
- Annotated for supervised learning.
- Preprocessed with resizing and normalization.

## 🛠️ Models Used

### 1. **Xception Network**
- Deep learning model with depthwise separable convolutions.
- Specialized for image classification tasks with high accuracy.

### 2. **Vision Transformers (ViT)**
- Employs transformer architecture for image classification.
- Processes images as sequences of patches.

### 3. **Efficient Network**
- Optimized for faster training and better resource usage.
- Balances accuracy and model size.

### 4. **Mobile Network**
- Lightweight architecture suitable for mobile and edge devices.
- Ensures lower inference time.

### 5. **Convolutional Neural Network (CNN)**
- Conventional approach with convolutional layers for feature extraction.
- Used as a baseline in this study.

## 🔬 Methodology

1. **Data Preprocessing**:
    - Resizing images to a uniform size.
    - Normalizing pixel values to [0, 1].
    - Splitting data into training, validation, and test sets.

2. **Model Training**:
    - Hyperparameter tuning for optimal performance.
    - Implementation using TensorFlow and PyTorch.

3. **Evaluation Metrics**:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - Computational efficiency (training/inference time).

4. **Visualization**:
    - Accuracy vs Epochs and Loss vs Epochs for training and validation data.
    - Confusion matrices.

## 🚀 Results

Key findings include:
- **Xception Network** achieved the highest accuracy, particularly in detecting subtle manipulations.
- **ViT** demonstrated strong performance in generalizing to unseen data but required significant computational resources.
- **Efficient Network** provided a good trade-off between speed and accuracy.
- **Mobile Network** was the fastest, making it suitable for deployment on resource-constrained devices.
- **CNN** showed decent performance but lagged behind other advanced models.

## 🧑‍💻 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow 2.x or PyTorch
- Required libraries: `numpy`, `matplotlib`, `scikit-learn`, `opencv-python`

### Installation
Clone this repository and install dependencies:

git clone https://github.com/yourusername/deepfake-classification.git
cd deepfake-classification
pip install -r requirements.txt

🤝 Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
