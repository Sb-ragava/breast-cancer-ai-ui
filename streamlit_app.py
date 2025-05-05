import streamlit as st
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
from fpdf import FPDF

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

# ✅ Preprocess for ResNet18 (Grad-CAM and IG)
def preprocess_for_resnet(pil_img):
    transform_tensor_resnet = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    tensor_img = transform_tensor_resnet(pil_img).unsqueeze(0)
    return tensor_img

# ✅ Custom CSS for left alignment
st.markdown("""
    <style>
        .css-1kyxreq { 
            text-align: left !important;
        }
        .css-1v3fvcr {
            justify-content: flex-start !important;
        }
        .stButton>button {
            text-align: left;
        }
        .stMarkdown {
            text-align: left !important;
        }
        .stFileUploader {
            text-align: left !important;
        }
    </style>
""", unsafe_allow_html=True)

# ✅ Page 1: Upload + Predict Unified
def page_1():
    st.title("Welcome to OncoAid")
    st.subheader("Your AI Assistant for Breast Cancer Detection and Explainability")

    st.write("OncoAid is an intelligent assistant designed to help detect breast cancer across multiple imaging modalities — Ultrasound, DDSM Mammography, and Histopathology.")
    st.write("It uses state-of-the-art AI models to classify tumors and provides visual explanations like Grad-CAM++ and Integrated Gradients to support clinical decision-making.")

    st.write("**Upload an image to get started and receive:**")
    st.write("- AI-based prediction")
    st.write("- Visual region importance maps")
    st.write("- A detailed case summary")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file

    if st.session_state.get("uploaded_file") is not None:
        if st.button("Predict"):
            # Clear previous prediction state to avoid corruption
            st.session_state.pred_class = None
            st.session_state.confidence = None
            st.session_state.probs = None
            st.session_state.raw_img_np = None
            st.session_state.input_tensor = None
            st.session_state.pred_idx = None
            st.session_state.pil_resized = None

            # Reset the prediction history if necessary
            if 'prediction_history' not in st.session_state:
                st.session_state.prediction_history = []

            # Preprocess and get prediction
            input_tensor, raw_img_np, pil_resized = preprocess_image(st.session_state.uploaded_file)

            with torch.no_grad():
                output = swin_model(input_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
                pred_idx = int(np.argmax(probs))
                pred_class = class_names[pred_idx]
                confidence = probs[pred_idx]

            # Store prediction history
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

            # Set state for display
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

                    for key in entry:
                        st.session_state[key] = entry[key]
                    st.session_state.page = 2
                    st.rerun()
            with col2:
                if st.button(f"Delete {len(st.session_state.prediction_history) - idx}"):

                    st.session_state.prediction_history.pop(len(st.session_state.prediction_history) - idx - 1)
                    st.rerun()

# ✅ Page 2: Prediction Results
def page_2():
    st.title(f"Prediction Results: {st.session_state.pred_class}")
    st.write(f"Confidence: {st.session_state.confidence*100:.2f}%")

    # Visualization of Grad-CAM++ (Example)
    cam = GradCAMPlusPlus(model=resnet_model, target_layers=[resnet_model.model.layer4[-1]])
    grayscale_cam = cam(input_tensor=st.session_state.input_tensor)[0, :]
    cam_image = show_cam_on_image(st.session_state.raw_img_np, grayscale_cam, use_rgb=True)
    
    # Integrated Gradients Visualization
    ig = IntegratedGradients(resnet_model)
    attribution = ig.attribute(st.session_state.input_tensor, target=st.session_state.pred_idx)
    attribution = attribution.squeeze().cpu().detach().numpy()
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    axs[0].imshow(cam_image)
    axs[0].set_title("Grad-CAM++ Visualization")
    
    axs[1].imshow(attribution, cmap="hot")
    axs[1].set_title("Integrated Gradients")
    
    st.pyplot(fig)

    st.markdown(""" 
### 🔍 Explanation of Grad-CAM++ and Integrated Gradients

#### Grad-CAM++:
- **Grad-CAM++** is a visualization technique that helps us understand which parts of the image are most important in making a decision.
- The highlighted areas in the image are where the model paid the most attention while making the prediction.
- In our case, areas showing **higher activation** help in determining if a tumor is benign or malignant.
- This technique enhances the **original Grad-CAM** method by providing more accurate localization in complex image scenarios.

#### Integrated Gradients (IG):
- **Integrated Gradients** (IG) is another technique for visualizing model decisions.
- It attributes the importance of each pixel in the image to the model's output by integrating the gradients of the output with respect to the input image.
- The **hot** regions in the IG visualization indicate pixels that contributed most to the prediction, helping us understand the **key features** the model is focusing on.
    """)

    st.markdown(""" 
### 🔍 Understanding the Prediction Distribution
After the image is analyzed by the AI model, the prediction isn't just a single label — it's a distribution of confidence across all possible classes: **Benign**, **Malignant**, and **Normal**. This helps provide transparency into how confident the model is in its decision.

#### 🟦 Bar Chart:
- The bar chart shows the exact **probability scores** for each class as predicted by the model.
- **Taller bars** indicate higher confidence.
- This is useful for comparing how close or far apart the predictions are.

#### 🕪 Pie Chart:
- The pie chart provides a **visual proportion** of the prediction confidence for each class.
- It helps quickly understand which class the model is leaning towards.
- The class with the **largest slice** is the model's top prediction.
    """)

if __name__ == "__main__":
    if 'page' not in st.session_state:
        st.session_state.page = 1

    if st.session_state.page == 1:
        page_1()
    elif st.session_state.page == 2:
        page_2()
