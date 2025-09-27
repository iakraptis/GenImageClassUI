import gradio as gr
import torch
from PIL import Image
from MultiClassifier import FakeImageClassifier, inference_transform

CLASS_NAMES = ["Real", "Sana 1.5", "SD 1.5", "SD 3.5"]

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

    with torch.no_grad():
        outputs = model(image)  # raw logits
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        print("Prediction probabilities:")
        for name, prob in zip(CLASS_NAMES, probs):
            print(f"  {name}: {prob*100:.2f}%")

        predicted_class = int(probs.argmax())
        predicted_percent = probs[predicted_class] * 100
        print(f"Predicted class: {CLASS_NAMES[predicted_class]} ({predicted_percent:.2f}%)")

    return predicted_class, probs

def predict_image_gradio(image_path):
    predicted_class, probs = predict_image(model, image_path, device)
    probability_lines = "\n".join(
        f"{name}: {prob*100:.2f}%" for name, prob in zip(CLASS_NAMES, probs)
    )
    return f"Predicted class: {CLASS_NAMES[predicted_class]}\n\nProbabilities:\n{probability_lines}"

# Set model path and device
model_path = "quadmodel.pth"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model_path, device)

iface = gr.Interface(
    fn=predict_image_gradio,
    inputs=gr.Image(type="filepath", label="Upload Image"),
    outputs=gr.Textbox(lines=10, label="Prediction"),
    title="Fake Image Classifier"
)

iface.launch(inbrowser=True)