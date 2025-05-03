import streamlit as st

# ✅ Must be first Streamlit command
st.set_page_config(page_title="OncoAid", layout="wide")

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import timm
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import gdown
from io import BytesIO
import base64

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import resnet18
from torchvision.models.swin_transformer import swin_t, Swin_T_Weights
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

# ✅ Class Labels
class_names = ['Benign', 'Malignant', 'Normal']

# ✅ Swin Transformer (for prediction)
class SwinClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(SwinClassifier, self).__init__()
        self.base_model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.base_model.head = nn.Linear(self.base_model.head.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# ✅ ResNet18 (for Grad-CAM++ + Integrated Gradients)
class ResNet18Visualizer(nn.Module):
    def __init__(self):
        super(ResNet18Visualizer, self).__init__()
        self.model = resnet18(pretrained=True)

    def forward(self, x):
        return self.model(x)

# ✅ Download model if not exists
swin_model_path = "swin_fusion_model.pth"
if not os.path.exists(swin_model_path):
    file_id = "1cOfU1mvbGNpt0gx2hGRzseoQMJXv7F6q"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, swin_model_path, quiet=False)

# ✅ Load models
swin_model = SwinClassifier()
swin_model.load_state_dict(torch.load(swin_model_path, map_location=torch.device('cpu')))
swin_model.eval()

resnet_model = ResNet18Visualizer()
resnet_model.eval()

# ✅ Utility to convert image to bytes
def image_to_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()

# ✅ Image Preprocessing
def preprocess_image(uploaded_file):
    pil_image = Image.open(uploaded_file).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor = preprocess(pil_image).unsqueeze(0)
    np_img = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255.0
    return tensor, np_img, pil_image.resize((224, 224))

# ✅ Download link utility
def get_image_download_link(img: Image.Image, filename: str = "explanation.png") -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">📥 Download Explanation Image</a>'
    return href

# ✅ Page 1: Upload Image
def page_1():
    st.title("👋 Welcome to OncoAid")
    st.markdown("""
    ## Your AI Assistant for Breast Cancer Detection and Explainability

    OncoAid is an intelligent assistant designed to help detect breast cancer across multiple imaging modalities — Ultrasound, DDSM Mammography, and Histopathology. It uses state-of-the-art AI models to classify tumors and provides visual explanations like Grad-CAM++ and Integrated Gradients to support clinical decision-making.

    **Upload an image to get started and receive:**
    ✅ AI-based prediction  
    ✅ Visual region importance maps  
    ✅ A detailed case summary
    """)

    uploaded_file = None
    with st.container():
        tab1, tab2 = st.tabs(["Upload", "Predict"])

        with tab1:
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
            st.session_state.uploaded_file = uploaded_file

        with tab2:
            if st.session_state.get("uploaded_file") is not None:
                st.image(st.session_state.uploaded_file, caption="Uploaded Image", use_column_width=True)
                if st.button("Predict"):
                    input_tensor, raw_img_np, pil_resized = preprocess_image(st.session_state.uploaded_file)

                    with torch.no_grad():
                        output = swin_model(input_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
                        pred_idx = int(np.argmax(probs))
                        pred_class = class_names[pred_idx]
                        confidence = probs[pred_idx]

                    if 'prediction_history' not in st.session_state:
                        st.session_state.prediction_history = []

                    st.session_state.prediction_history.append({
                        'pred_class': pred_class,
                        'confidence': confidence,
                        'probs': probs,
                        'raw_img_np': raw_img_np,
                        'input_tensor': input_tensor,
                        'pred_idx': pred_idx,
                        'pil_resized': pil_resized
                    })

                    if len(st.session_state.prediction_history) > 10:
                        st.session_state.prediction_history.pop(0)

                    st.session_state.pred_class = pred_class
                    st.session_state.confidence = confidence
                    st.session_state.probs = probs
                    st.session_state.raw_img_np = raw_img_np
                    st.session_state.input_tensor = input_tensor
                    st.session_state.pred_idx = pred_idx
                    st.session_state.pil_resized = pil_resized
                    st.session_state.page = 2
                    st.rerun()

    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("Prediction History")
        for idx, entry in enumerate(reversed(st.session_state.prediction_history)):
            st.markdown(f"**Prediction {len(st.session_state.prediction_history) - idx}:**")
            st.write(f"- Class: {entry['pred_class']}")
            st.write(f"- Confidence: {entry['confidence']*100:.2f}%")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(f"View {len(st.session_state.prediction_history) - idx}"):
                    st.session_state.pred_class = entry['pred_class']
                    st.session_state.confidence = entry['confidence']
                    st.session_state.probs = entry['probs']
                    st.session_state.raw_img_np = entry['raw_img_np']
                    st.session_state.input_tensor = entry['input_tensor']
                    st.session_state.pred_idx = entry['pred_idx']
                    st.session_state.pil_resized = entry['pil_resized']
                    st.session_state.page = 2
                    st.rerun()
            with col2:
                if st.button(f"Delete {len(st.session_state.prediction_history) - idx}"):
                    st.session_state.prediction_history.pop(len(st.session_state.prediction_history) - idx - 1)
                    st.rerun()

# ✅ Page 2: Results and Explainability

def page_2():
    st.title("🧠 Prediction & Explainability")
    st.write(f"### Prediction: {st.session_state.pred_class} ({st.session_state.confidence*100:.2f}%)")

    col1, col2 = st.columns(2)

    # Grad-CAM++
    target_layers = [resnet_model.model.layer4[-1]]
    cam = GradCAMPlusPlus(model=resnet_model.model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=st.session_state.input_tensor, targets=[ClassifierOutputTarget(st.session_state.pred_idx)])[0]
    visualization = show_cam_on_image(st.session_state.raw_img_np, grayscale_cam, use_rgb=True)
    cam_img = Image.fromarray(visualization)
    col1.image(cam_img, caption="Grad-CAM++", use_column_width=True)
    col1.markdown(get_image_download_link(cam_img, filename="grad_cam.png"), unsafe_allow_html=True)

    # Integrated Gradients
    ig = IntegratedGradients(resnet_model)
    attr = ig.attribute(st.session_state.input_tensor, target=st.session_state.pred_idx, n_steps=50)
    attr = attr.squeeze().cpu().detach().numpy()
    attr = np.transpose(attr, (1, 2, 0))
    attr = (attr - attr.min()) / (attr.max() - attr.min())
    ig_img = Image.fromarray(np.uint8(attr * 255))
    col2.image(ig_img, caption="Integrated Gradients", use_column_width=True)
    col2.markdown(get_image_download_link(ig_img, filename="integrated_gradients.png"), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Case Summary")
    st.write(f"**Prediction Class**: {st.session_state.pred_class}")
    st.write(f"**Confidence**: {st.session_state.confidence*100:.2f}%")
    st.write("**Class Probabilities:**")
    for i, c in enumerate(class_names):
        st.write(f"- {c}: {st.session_state.probs[i]*100:.2f}%")

    if st.button("🔙 Back"):
        st.session_state.page = 1
        st.rerun()

# ✅ Main entry point

def main():
    if 'page' not in st.session_state:
        st.session_state.page = 1

    if st.session_state.page == 1:
        page_1()
    elif st.session_state.page == 2:
        page_2()

if __name__ == "__main__":
    main()
