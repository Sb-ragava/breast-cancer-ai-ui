import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # Use PIL for image handling
import timm
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import gdown
import os

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
    gdown.download("https://drive.google.com/uc?id=1K60lvI-NDOWVCRzUekSLjZmQXqBoqKcn", swin_model_path, quiet=False)

# ✅ Load models
swin_model = SwinClassifier()
swin_model.load_state_dict(torch.load(swin_model_path, map_location=torch.device('cpu')))
swin_model.eval()

resnet_model = ResNet18Visualizer()
resnet_model.eval()

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

# ✅ Replace cv2-based scaling function with PIL-based one
def scale_cam_image(cam_image: np.ndarray, img_size: tuple):
    pil_img = Image.fromarray(cam_image)
    pil_img = pil_img.resize(img_size, Image.ANTIALIAS)
    return np.array(pil_img)

# ✅ Modify other necessary parts where cv2 was previously used
def deprocess_image(img: np.ndarray):
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img

# ✅ Page 1: Upload Image
def page_1():
    st.title("Welcome to the Image Classification App!")
    st.write("Upload an image for classification.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        input_tensor, raw_img_np, pil_resized = preprocess_image(uploaded_file)

        with torch.no_grad():
            output = swin_model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_class = class_names[pred_idx]
            confidence = probs[pred_idx]

        if st.button("Predict"):
            st.session_state.pred_class = pred_class
            st.session_state.confidence = confidence
            st.session_state.probs = probs
            st.session_state.raw_img_np = raw_img_np
            st.session_state.input_tensor = input_tensor
            st.session_state.pred_idx = pred_idx
            st.experimental_rerun()

# ✅ Page 2: Prediction Results
def page_2():
    st.title(f"Prediction Results: {st.session_state.pred_class}")

    st.write(f"Confidence: {st.session_state.confidence*100:.2f}%")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    sns.barplot(x=class_names, y=st.session_state.probs, palette='coolwarm')
    plt.title("Prediction Probabilities (Bar Chart)")
    plt.ylabel("Probability")

    plt.subplot(1, 2, 2)
    plt.pie(st.session_state.probs, labels=class_names, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('coolwarm'))
    plt.title("Prediction Confidence (Pie Chart)")

    plt.tight_layout()
    st.pyplot()

    target_layers = [resnet_model.model.layer4[-1]]
    cam = GradCAMPlusPlus(model=resnet_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=st.session_state.input_tensor, targets=[ClassifierOutputTarget(st.session_state.pred_idx)])[0]
    visualization = show_cam_on_image(st.session_state.raw_img_np, grayscale_cam, use_rgb=True)

    st.image(visualization, caption="Grad-CAM++ Explanation", use_column_width=True)

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

    st.image(heatmap, caption="Integrated Gradients Explanation", use_column_width=True)

    st.write("### Prediction Summary:")
    st.write(f"1. The model predicts that this image belongs to the '{st.session_state.pred_class}' class.")
    st.write(f"2. Confidence of the prediction: {st.session_state.confidence*100:.2f}%")
    st.write(f"3. Key regions of the image were highlighted using Grad-CAM++.")
    st.write(f"4. Integrated Gradients shows which pixels contributed most to the prediction.")

    st.download_button(
        label="Download Grad-CAM++ Image",
        data=Image.fromarray(visualization),
        file_name="grad_cam_image.png",
        mime="image/png"
    )

    st.download_button(
        label="Download IG Image",
        data=Image.fromarray((heatmap.squeeze() * 255).astype(np.uint8)),
        file_name="ig_image.png",
        mime="image/png"
    )

# ✅ Main function to control pages
def main():
    st.set_page_config(page_title="Streamlit Multi-Page App")

    if 'pred_class' not in st.session_state:
        st.session_state.pred_class = None

    page = st.sidebar.radio("Choose a page", ("Home", "Prediction Results"))

    if page == "Home":
        page_1()
    elif page == "Prediction Results" and st.session_state.pred_class is not None:
        page_2()
    else:
        st.write("Please upload an image to make a prediction.")

if __name__ == "__main__":
    main()
