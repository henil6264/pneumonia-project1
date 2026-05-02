import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(
        torch.load("model/pneumonia_model.pth", map_location="cpu")
    )
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # X-rays are grayscale
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
    labels = ["Normal", "Pneumonia"]
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray"),
    outputs=gr.Label(num_top_classes=2, label="Result"),
    title="🫁 Pneumonia Detection",
    description="Upload a chest X-ray. The model will predict Normal or Pneumonia.",
    theme=gr.themes.Soft()
)

demo.launch()
