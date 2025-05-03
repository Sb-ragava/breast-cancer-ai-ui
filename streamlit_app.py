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

        with tab2:
            if uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                if st.button("Predict"):
                    input_tensor, raw_img_np, pil_resized = preprocess_image(uploaded_file)

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
        with st.expander("Previous Predictions", expanded=True):
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

# ✅ Remaining code: page_2(), preprocess_image(), and main()
# Add them below if needed


# ✅ Page 2: Prediction Results
def page_2():
    st.title(f"Prediction Results: {st.session_state.pred_class}")

    st.write(f"Confidence: {st.session_state.confidence*100:.2f}%")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    sns.barplot(x=class_names, y=st.session_state.probs, hue=class_names, palette='coolwarm', legend=False, ax=axs[0])
    axs[0].set_title("Prediction Probabilities (Bar Chart)")
    axs[0].set_ylabel("Probability")

    axs[1].pie(
        st.session_state.probs,
        labels=class_names,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette('coolwarm')
    )
    axs[1].set_title("Prediction Confidence (Pie Chart)")

    st.pyplot(fig)

    target_layers = [resnet_model.model.layer4[-1]]
    cam = GradCAMPlusPlus(model=resnet_model, target_layers=target_layers)
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

    st.write("### Explainability Visualizations")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(st.session_state.pil_resized, caption="Original Image", use_container_width=True)
    with col2:
        st.image(visualization, caption="Grad-CAM++", use_container_width=True)
    with col3:
        st.image(heatmap, caption="Integrated Gradients", use_container_width=True)

    st.write("### Prediction Summary:")
    st.write(f"1. The model predicts that this image belongs to the '{st.session_state.pred_class}' class.")
    st.write(f"2. Confidence of the prediction: {st.session_state.confidence*100:.2f}%")
    st.write(f"3. Key regions of the image were highlighted using Grad-CAM++.")
    st.write(f"4. Integrated Gradients shows which pixels contributed most to the prediction.")

    gradcam_pil = Image.fromarray(visualization)
    ig_pil = Image.fromarray((heatmap.squeeze() * 255).astype(np.uint8))

    st.download_button(
        label="Download Grad-CAM++ Image",
        data=image_to_bytes(gradcam_pil),
        file_name="grad_cam_image.png",
        mime="image/png"
    )

    st.download_button(
        label="Download IG Image",
        data=image_to_bytes(ig_pil),
        file_name="ig_image.png",
        mime="image/png"
    )

    if st.button("Back"):
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        st.session_state.prediction_history.append({
            "pred_class": st.session_state.pred_class,
            "confidence": st.session_state.confidence,
            "probs": st.session_state.probs,
            "raw_img_np": st.session_state.raw_img_np,
            "input_tensor": st.session_state.input_tensor,
            "pred_idx": st.session_state.pred_idx,
            "pil_resized": st.session_state.pil_resized
        })
        if len(st.session_state.prediction_history) > 10:
            st.session_state.prediction_history.pop(0)
        for key in ["pred_class", "confidence", "probs", "raw_img_np", "input_tensor", "pred_idx", "pil_resized"]:
            st.session_state.pop(key, None)
        st.session_state.page = 1
        st.rerun()

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

# ✅ Main function to control pages
def main():
    st.set_page_config(page_title="Streamlit Multi-Page App")

    if 'page' not in st.session_state:
        st.session_state.page = 1
        st.session_state.pred_class = None

    if st.session_state.page == 1:
        page_1()
    elif st.session_state.page == 2:
        page_2()

if __name__ == "__main__":
    main()
