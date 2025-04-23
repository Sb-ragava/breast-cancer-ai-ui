# streamlit_app.py
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import pandas as pd
from lime import lime_image
from skimage.segmentation import mark_boundaries
from captum.attr import IntegratedGradients
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import gdown

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Breast Cancer AI Classifier", layout="wide")

# ---------------------- MODEL ----------------------
class SwinClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(SwinClassifier, self).__init__()
        from torchvision.models import swin_t
        self.base_model = swin_t(weights='IMAGENET1K_V1')
        self.base_model.head = nn.Linear(self.base_model.head.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

@st.cache_resource
def load_model():
    model = SwinClassifier()
    model_url = "https://drive.google.com/uc?id=1cOfU1mvbGNpt0gx2hGRzseoQMJXv7F6q"
    output_path = "swin_model.pth"

    try:
        if not os.path.exists(output_path):
            with st.spinner("🔄 Downloading model from Google Drive..."):
                gdown.download(model_url, output_path, quiet=False)

        with st.spinner("🔧 Loading model..."):
            model.load_state_dict(torch.load(output_path, map_location=torch.device('cpu')))
            model.eval()
            st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")

    return model

model = load_model()
class_names = ['Benign_Processed', 'Malignant_Processed', 'Normal_Processed']
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------- EXPLAINABILITY ----------------------
def generate_gradcam(image_tensor, raw_image):
    cam = GradCAMPlusPlus(model=model, target_layers=[model.base_model.features[-1]])
    grayscale_cam = cam(input_tensor=image_tensor, targets=None)[0]
    grayscale_cam = np.where(grayscale_cam > 0.3, grayscale_cam, 0)
    cam_image = show_cam_on_image(raw_image, grayscale_cam, use_rgb=True, image_weight=0.7)
    return cam_image

def generate_lime(image_np, pil_img):
    explainer = lime_image.LimeImageExplainer()
    def batch_predict(images):
        batch = torch.stack([transform(Image.fromarray(img)) for img in images], dim=0)
        logits = model(batch)
        return torch.softmax(logits, dim=1).detach().numpy()
    explanation = explainer.explain_instance(image_np, batch_predict, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
    return mark_boundaries(temp / 255.0, mask)

def generate_ig(image_tensor):
    ig = IntegratedGradients(model)
    image_tensor.requires_grad_()
    attr_ig, _ = ig.attribute(image_tensor, target=0, return_convergence_delta=True)
    attr_ig = attr_ig.squeeze().detach().numpy().transpose(1, 2, 0)
    attr_ig = (attr_ig - attr_ig.min()) / (attr_ig.max() - attr_ig.min())
    return attr_ig

# ---------------------- PDF REPORT ----------------------
def generate_pdf(image_path, prediction, confidences):
    report_path = os.path.join(tempfile.gettempdir(), "breast_cancer_report.pdf")
    c = canvas.Canvas(report_path, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(30, 750, "Breast Cancer Classification Report")
    c.setFont("Helvetica", 12)
    c.drawString(30, 730, f"Prediction: {prediction}")
    c.drawString(30, 710, "Confidence Scores:")

    y = 690
    for label, score in confidences.items():
        c.drawString(40, y, f"{label}: {score:.2f}%")
        y -= 20

    c.drawString(30, y - 20, "Recommendation:")
    if prediction != "Normal_Processed":
        c.drawString(40, y - 40, "⚠️ Please consult a doctor immediately.")
    else:
        c.drawString(40, y - 40, "✅ No signs detected. Still, regular checkups are advised.")

    c.showPage()
    c.save()
    return report_path

# ---------------------- UI ----------------------
st.markdown("""
    <h1 style='text-align: center; color: darkblue;'>🧬 Breast Cancer Diagnosis Assistant</h1>
    <p style='text-align: center;'>Powered by Swin Transformer + Explainable AI (Grad-CAM++, LIME, IG)</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a medical image (.png or .jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Original Image", use_column_width=True)

    image_tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0].numpy()
        pred_idx = np.argmax(probs)
        pred_label = class_names[pred_idx]

    with col2:
        st.subheader("🩺 Prediction Result")
        if pred_label == "Malignant_Processed":
            st.error("⚠️ Predicted: Malignant - Immediate Consultation Recommended")
        elif pred_label == "Benign_Processed":
            st.warning("🟡 Predicted: Benign - Follow-up Required")
        else:
            st.success("✅ Predicted: Normal - No signs detected")
        st.markdown(f"**Model Accuracy:** 98.31%")

    st.markdown("### 📈 Confidence Scores")
    conf_df = pd.DataFrame({
        'Class': class_names,
        'Confidence (%)': [round(p * 100, 2) for p in probs]
    })
    st.dataframe(conf_df)

    raw_img = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0

    st.markdown("### 🔍 Explainability Visualizations")
    exp_cols = st.columns(3)

    with st.spinner("🔴 Generating Grad-CAM++..."):
        gradcam_img = generate_gradcam(image_tensor, raw_img)
        exp_cols[0].image(gradcam_img, caption="Grad-CAM++", use_column_width=True)

    with st.spinner("🟢 Generating LIME explanation..."):
        lime_img = generate_lime(np.array(pil_img.resize((224, 224))), pil_img)
        exp_cols[1].image(lime_img, caption="LIME", use_column_width=True)

    with st.spinner("🔵 Generating Integrated Gradients..."):
        ig_img = generate_ig(image_tensor)
        exp_cols[2].image(ig_img, caption="Integrated Gradients", use_column_width=True)

    with st.expander("🔎 How Explainability Works"):
        st.markdown("""
        - **Grad-CAM++** highlights class-relevant regions in feature maps.
        - **LIME** breaks the image into superpixels to explain predictions locally.
        - **Integrated Gradients** attributes each pixel's importance using gradients.
        """)

    st.markdown("### 📄 Interpretation")
    if pred_label != "Normal_Processed":
        st.warning("⚠️ This image shows signs of a malignant or benign tumor. Please consult a doctor for further examination.")
    else:
        st.success("✅ The image shows no signs of cancer. Regular screenings are still recommended.")

    if st.button("📥 Download PDF Report"):
        pdf_path = generate_pdf(
            uploaded_file.name,
            pred_label.replace('_Processed',''),
            {class_names[i]: probs[i]*100 for i in range(len(class_names))}
        )
        with open(pdf_path, "rb") as f:
            st.download_button(label="Download Report", data=f, file_name="breast_cancer_report.pdf", mime="application/pdf")
