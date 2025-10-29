import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Skin Disease Classifier", page_icon="🏥", layout="wide")

DISEASE_CLASSES = {
    0: "Melanoma",
    1: "Nevus",
    2: "Basal Cell Carcinoma",
    3: "Actinic Keratosis",
    4: "Benign Keratosis",
    5: "Dermatofibroma",
    6: "Vascular Lesion"
}

DISEASE_DESCRIPTIONS = {
    "Melanoma": "A serious form of skin cancer. Requires immediate medical attention.",
    "Nevus": "A common mole. Usually benign but should be monitored.",
    "Basal Cell Carcinoma": "The most common type of skin cancer. Generally treatable.",
    "Actinic Keratosis": "A precancerous lesion caused by sun exposure.",
    "Benign Keratosis": "A common, non-cancerous skin growth.",
    "Dermatofibroma": "A benign skin nodule, usually harmless.",
    "Vascular Lesion": "An abnormal collection of blood vessels."
}

@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(DISEASE_CLASSES))
    model.eval()
    return model

def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    return transform(image).unsqueeze(0)

def predict_disease(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    top_probs, top_indices = torch.topk(probabilities[0], 3)
    predictions = {
        "primary_disease": DISEASE_CLASSES[predicted_class.item()],
        "confidence": confidence.item(),
        "top_3": [
            {"disease": DISEASE_CLASSES[idx.item()], "confidence": prob.item()}
            for idx, prob in zip(top_indices, top_probs)
        ]
    }
    return predictions

def get_confidence_color(confidence):
    if confidence >= 0.8:
        return "color: green; font-weight: bold;"
    elif confidence >= 0.6:
        return "color: orange; font-weight: bold;"
    else:
        return "color: red; font-weight: bold;"

st.title("🏥 Dermatoscopic Skin Disease Classifier")
st.markdown("Upload a dermatoscopic skin image (JPG/PNG) for classification.")

model = load_model()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_tensor = preprocess_image(image)
    predictions = predict_disease(model, image_tensor)

    st.subheader("Prediction")
    confidence_color = get_confidence_color(predictions["confidence"])
    st.markdown(f"**Disease:** {predictions['primary_disease']}")
    st.markdown(f"<span style='{confidence_color}'>Confidence: {predictions['confidence']:.2%}</span>", unsafe_allow_html=True)
    
    st.info(DISEASE_DESCRIPTIONS[predictions["primary_disease"]])

    st.subheader("Top 3 Predictions")
    for i, pred in enumerate(predictions["top_3"], 1):
        st.write(f"{i}. {pred['disease']} - {pred['confidence']:.2%}")

    st.warning("⚠️ This tool is for educational purposes only and is not a substitute for professional medical advice.")
else:
    st.info("Please upload an image to get started.")
