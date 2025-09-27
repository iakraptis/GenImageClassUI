import gradio as gr
import pathlib
import torch
from torchvision import transforms
from PIL import Image
from MultiClassifier import FakeImageClassifier, inference_transform

def load_model(model_path, device):
    model = FakeImageClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = inference_transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        rounded_outputs = torch.round(outputs, decimals=2)

        print(rounded_outputs)
        predicted_class = torch.argmax(outputs, dim=1).item()

    return predicted_class


# Set model path and device
model_path = "path/to/your/model.pth"  # <-- Update this path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model_path, device)

iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    title="Fake Image Classifier"
)

iface.launch()