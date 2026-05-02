import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import os

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = "model/pneumonia_model.pth"

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

# ----------------------------
# Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------------------
# UI
# ----------------------------
st.title("🫁 Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image and the model will predict whether it shows **Normal** lungs or **Pneumonia**.")

st.markdown("---")

if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model file not found at `model/pneumonia_model.pth`. Please train the model first by running `python src/train.py` or place the trained model file in the `model/` folder.")
    st.stop()

model, device = load_model()

uploaded_file = st.file_uploader("📤 Upload a Chest X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    img = transform(image)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)

    classes = ["Normal", "Pneumonia"]
    result = classes[pred.item()]
    confidence = probabilities[0][pred.item()].item() * 100

    st.markdown("---")
    st.subheader("🔍 Prediction Result:")

    if result == "Pneumonia":
        st.error(f"🔴 Result: **{result}**  \nConfidence: `{confidence:.2f}%`")
        st.warning("⚠️ Please consult a medical professional for proper diagnosis.")
    else:
        st.success(f"🟢 Result: **{result}**  \nConfidence: `{confidence:.2f}%`")

    st.markdown("---")
    st.caption("⚠️ This tool is for educational purposes only and is not a substitute for medical advice.")
