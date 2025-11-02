# 🌿 GreenLens AI - Intelligent Waste Classifier

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-70%25-green)
![Status](https://img.shields.io/badge/Status-Week%201%20Complete-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

<div align="center">
  <h3>🔍 Computer Vision for Sustainable Waste Management</h3>
  <p><i>Using AI to see waste differently, one image at a time</i></p>
</div>

---

## 🎯 About GreenLens AI

**GreenLens AI** is an intelligent waste classification system that leverages deep learning and computer vision to automatically categorize waste into 12 distinct types. By combining Convolutional Neural Networks (CNN) with image recognition technology, GreenLens AI promotes sustainable waste management and efficient recycling practices.

### 🌟 Mission
*To make waste segregation smarter, faster, and more accurate through the power of artificial intelligence.*

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **AI-Powered** | Advanced CNN architecture for accurate classification |
| 👁️ **Computer Vision** | Real-time image processing and recognition |
| ⚡ **Fast Inference** | Predictions in under 2 seconds |
| 📊 **High Accuracy** | 70%+ validation accuracy baseline |
| 🌱 **Eco-Friendly** | Promotes proper waste segregation |
| 🔓 **Open Source** | Free to use and contribute |

---

## 📊 Dataset Overview

- **Source**: Kaggle - Garbage Classification Dataset
- **Total Images**: ~15,000 high-quality images
- **Training Split**: 80% (12,000 images)
- **Validation Split**: 20% (3,000 images)

### 📦 Waste Categories (12 Classes)
```
🔋 battery          🌱 biological       🟤 brown-glass
📦 cardboard        👕 clothes          🟢 green-glass
🔩 metal            📄 paper            🥤 plastic
👟 shoes            🗑️  trash            ⚪ white-glass
```

---

## 🧠 Model Architecture

### Deep Learning Configuration
```python
Architecture: Custom Convolutional Neural Network (CNN)
Input Shape:  224 × 224 × 3 (RGB)
Total Params: ~10M parameters

Layers:
├── Conv2D Block 1: 32 filters  → MaxPool → BatchNorm
├── Conv2D Block 2: 64 filters  → MaxPool → BatchNorm
├── Conv2D Block 3: 128 filters → MaxPool → BatchNorm
├── Conv2D Block 4: 256 filters → MaxPool → BatchNorm
├── Flatten
├── Dense: 512 units → Dropout(0.5)
├── Dense: 256 units → Dropout(0.3)
└── Output: 12 classes (Softmax)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam (lr=0.001) |
| **Loss Function** | Categorical Crossentropy |
| **Batch Size** | 32 |
| **Epochs** | 20 |
| **Data Augmentation** | Rotation, Shift, Zoom, Flip |
| **Callbacks** | EarlyStopping, ReduceLROnPlateau |

---

## 📈 Performance Metrics

### Week 1 Results (Baseline Model)

| Metric | Score |
|--------|-------|
| **Training Accuracy** | 73.2% |
| **Validation Accuracy** | 70.5% |
| **Training Loss** | 0.821 |
| **Validation Loss** | 0.943 |
| **Training Time** | ~25 minutes (GPU) |
| **Model Size** | 48.6 MB |
| **Inference Time** | <2 seconds/image |

### 📊 Visualizations

- ✅ Training/Validation accuracy curves
- ✅ Loss convergence plots
- ✅ Confusion matrix (12×12)
- ✅ Per-class accuracy breakdown
- ✅ Sample predictions with confidence scores

---

## 🛠️ Technology Stack

### Core Technologies
```
Language:       Python 3.10
Framework:      TensorFlow 2.15.0
Backend:        Keras
Environment:    Google Colab (GPU T4)
```

### Libraries & Tools
```python
# Deep Learning
tensorflow==2.15.0
keras==2.15.0

# Data Processing
numpy==1.24.3
pandas==2.0.3
opencv-python==4.8.0

# Visualization
matplotlib==3.7.2
seaborn==0.12.2

# Machine Learning
scikit-learn==1.3.0

# Utilities
pillow==10.0.0
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Google Colab account (recommended)
- Kaggle API credentials

### Installation & Setup

#### 1️⃣ Clone Repository
```bash
git clone https://github.com/saileed05/GreenLens-AI.git
cd GreenLens-AI
```

#### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3️⃣ Setup Kaggle API
```bash
# Download kaggle.json from Kaggle.com → Account → API
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### 4️⃣ Run in Google Colab
1. Open `greenlens_week1.ipynb` in Google Colab
2. **Runtime** → **Change runtime type** → **GPU**
3. Upload `kaggle.json` when prompted
4. Run all cells

#### 5️⃣ Dataset Auto-Download
The notebook automatically downloads the dataset from Kaggle.

---

## 📁 Project Structure
```
GreenLens-AI/
│
├── 📓 greenlens_week1.ipynb          # Main training notebook
├── 📋 requirements.txt                # Python dependencies
├── 📖 README.md                       # Project documentation
│
├── 📊 results/                        # Training results
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   └── sample_predictions.png
│
├── 🔧 utils/                          # Utility scripts (future)
│   ├── preprocessing.py
│   └── visualization.py
│
└── 🤖 models/                         # Saved models (future)
    └── greenlens_v1.h5
```

## 🌍 Environmental Impact

### Why GreenLens AI Matters

| Impact Area | Benefit |
|-------------|---------|
| ♻️ **Recycling** | Automates waste sorting for efficient recycling |
| 🌱 **Sustainability** | Reduces landfill waste and pollution |
| 🤖 **Efficiency** | Eliminates human error in waste classification |
| 📊 **Data Insights** | Provides analytics for waste management |
| 🌏 **Scalability** | Deployable in smart cities worldwide |
| 💰 **Cost Savings** | Reduces manual sorting labor costs |

### Real-World Applications
- 🏢 Smart waste bins in offices
- 🏙️ Municipal waste management systems
- 🏭 Industrial waste sorting facilities
- 🏠 Home recycling assistants
- 📱 Mobile waste classification apps

---


## 📚 Key Learnings

### Technical Skills Developed
- ✅ Designing CNN architectures for image classification
- ✅ Implementing data augmentation techniques
- ✅ Handling multi-class imbalanced datasets
- ✅ Model evaluation and performance metrics
- ✅ Working with large-scale image datasets
- ✅ Optimizing training with callbacks

### Domain Knowledge Gained
- 🌍 Understanding waste management challenges
- ♻️ Learning recycling best practices
- 📊 Data-driven sustainability solutions
- 🤖 AI applications in environmental protection

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 **Fork** the repository
2. 🌟 **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 **Push** to the branch (`git push origin feature/AmazingFeature`)
5. 🔀 **Open** a Pull Request

### Areas for Contribution
- 🐛 Bug fixes and improvements
- 📝 Documentation enhancements
- 🎨 UI/UX improvements
- 🧪 Additional model architectures
- 📊 Data collection and annotation
- 🌍 Translations

---

## 📧 Contact & Support

**Developer**: Saileed  
**GitHub**: [@saileed05](https://github.com/saileed05)  
**Project Link**: [GreenLens-AI](https://github.com/saileed05/GreenLens-AI)

---

## 🙏 Acknowledgments

- **Dataset**: Kaggle Community for the Garbage Classification dataset
- **Framework**: TensorFlow and Keras teams
- **Infrastructure**: Google Colab for free GPU access
- **Inspiration**: Global sustainability and environmental goals
- **Community**: Open-source ML community

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
MIT License - Free for personal and commercial use
```

---

## 🌟 Show Your Support

If you find **GreenLens AI** helpful, please consider:

- ⭐ **Starring** this repository
- 🍴 **Forking** for your own experiments
- 📢 **Sharing** with your network
- 💬 **Providing feedback** through issues

---

<div align="center">

### 🌿 Together, let's build a sustainable future with AI! 🤖

![Green Divider](https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif)



*Last Updated: November 2024 | Version: 1.0.0 (Week 1)*

</div>
