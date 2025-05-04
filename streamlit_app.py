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

# ✅ Preprocess image function
def preprocess_image(img_file):
    pil_img = Image.open(img_file).convert('RGB')
    transform_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor_img = transform_tensor(pil_img).unsqueeze(0)
    raw_img_np = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
    return tensor_img, raw_img_np, pil_img.resize((224, 224))

# ✅ Page 1: Upload + Predict Unified

def page_1():
    st.markdown("<h1 style='text-align: center;'>OncoAid</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center;'>
        <h4>Your AI Assistant for Breast Cancer Detection</h4>
        <p>Upload an image to get predictions and visual explanations using state-of-the-art AI models.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    uploaded_file = st.file_uploader("Upload an image (jpg/png/jpeg)...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        input_tensor, raw_img_np, pil_resized = preprocess_image(uploaded_file)
        st.image(pil_resized, caption="Uploaded Image", use_container_width=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Predict"):
                with torch.no_grad():
                    output = swin_model(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
                    pred_idx = int(np.argmax(probs))
                    pred_class = class_names[pred_idx]
                    confidence = probs[pred_idx]

                st.session_state.pred_class = pred_class
                st.session_state.confidence = confidence
                st.session_state.probs = probs
                st.session_state.raw_img_np = raw_img_np
                st.session_state.input_tensor = input_tensor
                st.session_state.pred_idx = pred_idx
                st.session_state.pil_resized = pil_resized

                st.session_state.page = 2
                st.rerun()
        with col2:
            if st.button("Clear"):
                st.session_state.clear()
                st.rerun()

# ✅ Page 2: Prediction Results
def page_2():
    st.markdown(f"<h2>Prediction: {st.session_state.pred_class} ({st.session_state.confidence*100:.2f}%)</h2>", unsafe_allow_html=True)

    target_layers = [resnet_model.model.layer4[-1]]
    cam = GradCAMPlusPlus(model=resnet_model.model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=st.session_state.input_tensor, targets=[ClassifierOutputTarget(st.session_state.pred_idx)])[0]
    visualization = show_cam_on_image(st.session_state.raw_img_np, grayscale_cam, use_rgb=True)

    ig = IntegratedGradients(resnet_model)
    st.session_state.input_tensor.requires_grad_()
    baseline = torch.zeros_like(st.session_state.input_tensor)
    attributions_ig = ig.attribute(inputs=st.session_state.input_tensor,
                                   baselines=baseline,
                                   target=st.session_state.pred_idx,
                                   n_steps=50)

    attr_ig_np = attributions_ig.squeeze().detach().numpy()
    input_np = st.session_state.input_tensor.squeeze().permute(1, 2, 0).detach().numpy()
    input_np = (input_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    input_np = np.clip(input_np, 0, 1)

    heatmap = np.sum(attr_ig_np, axis=0, keepdims=True)
    heatmap = np.transpose(heatmap, (1, 2, 0))
    heatmap = np.clip(heatmap, 0, 1)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(st.session_state.pil_resized, caption="Original Image", use_container_width=True)
    with col2:
        st.image(visualization, caption="Grad-CAM++", use_container_width=True)
    with col3:
        st.image(heatmap, caption="Integrated Gradients", use_container_width=True)

    gradcam_pil = Image.fromarray(visualization)
    ig_pil = Image.fromarray((heatmap.squeeze() * 255).astype(np.uint8))

    col4, col5 = st.columns([1, 1])
    with col4:
        st.download_button("Download Grad-CAM++", image_to_bytes(gradcam_pil), "grad_cam.png", mime="image/png")
    with col5:
        st.download_button("Download IG", image_to_bytes(ig_pil), "integrated_gradients.png", mime="image/png")

    st.write("### Summary:")
    st.write(f"- **Prediction:** {st.session_state.pred_class}")
    st.write(f"- **Confidence:** {st.session_state.confidence*100:.2f}%")
    st.write("- Grad-CAM++ highlights important regions in the image.")
    st.write("- Integrated Gradients shows which pixels most contributed to the prediction.")

    if st.button("← Back"):
        st.session_state.page = 1
        st.rerun()

# ✅ Main function to control pages
def main():
    if 'page' not in st.session_state:
        st.session_state.page = 1

    if st.session_state.page == 1:
        page_1()
    elif st.session_state.page == 2:
        page_2()

if __name__ == "__main__":
    main()
