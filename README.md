title: Pneumonia Detection
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
# 🫁 Pneumonia Detection from Chest X-Ray

A deep learning web app that detects **Pneumonia** from chest X-ray images using a fine-tuned ResNet18 model, deployed with Streamlit.

---

## ✨ Features

- Upload chest X-ray images (JPG/PNG)
- Predicts **Normal** or **Pneumonia**
- Shows confidence score
- Real-time inference
- Clean UI using Streamlit

---

## 🧠 Model Details

| Detail | Info |
|--------|------|
| Architecture | ResNet18 |
| Training | Transfer Learning (ImageNet weights) |
| Classes | Normal, Pneumonia |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam (lr=0.001) |
| Epochs | 5 |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~95% |
| Pneumonia Recall | ~96% |

> In medical AI, minimizing false negatives is critical. This model achieves ~96% recall on Pneumonia cases.

---

## 🚀 Installation & Running Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/pneumonia-detection.git
cd pneumonia-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your trained model
# Place pneumonia_model.pth inside the model/ folder

# 4. Run the app
streamlit run app.py
```

---

## 🗂️ Project Structure

```
pneumonia-detection/
├── app.py                  # Streamlit web app
├── requirements.txt        # Python dependencies
├── model/
│   └── pneumonia_model.pth # Trained model (not included, see model/README.md)
├── src/
│   ├── model.py            # ResNet18 model definition
│   ├── data_loader.py      # Dataset loading & augmentation
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation & metrics
│   └── test_data.py        # Quick data check script
└── data/                   # Dataset folder (not included)
    └── train/
        ├── NORMAL/
        └── PNEUMONIA/
```

---

## 📦 Dataset

Chest X-Ray dataset from Kaggle:  
[https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

> Dataset not included in this repo due to size.

---

## 🔮 Future Improvements

- Add Grad-CAM heatmap visualization
- Reduce false negatives further
- Add confidence threshold warnings
- Deploy on Hugging Face Spaces

---

## 👤 Author

Made with ❤️ using PyTorch + Streamlit
