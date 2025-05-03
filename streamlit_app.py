import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import timm
import seaborn as sns
from sklearn.metrics import classification\_report, confusion\_matrix
import os
import gdown
from io import BytesIO

from pytorch\_grad\_cam import GradCAMPlusPlus
from pytorch\_grad\_cam.utils.image import show\_cam\_on\_image
from pytorch\_grad\_cam.utils.model\_targets import ClassifierOutputTarget
from torchvision.models import resnet18
from torchvision.models.swin\_transformer import swin\_t, Swin\_T\_Weights
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

# ✅ Class Labels

class\_names = \['Benign', 'Malignant', 'Normal']

# ✅ Swin Transformer (for prediction)

class SwinClassifier(nn.Module):
def **init**(self, num\_classes=3):
super(SwinClassifier, self).**init**()
self.base\_model = swin\_t(weights=Swin\_T\_Weights.IMAGENET1K\_V1)
self.base\_model.head = nn.Linear(self.base\_model.head.in\_features, num\_classes)

```
def forward(self, x):
    return self.base_model(x)
```

# ✅ ResNet18 (for Grad-CAM++ + Integrated Gradients)

class ResNet18Visualizer(nn.Module):
def **init**(self):
super(ResNet18Visualizer, self).**init**()
self.model = resnet18(pretrained=True)

```
def forward(self, x):
    return self.model(x)
```

# ✅ Download model if not exists

swin\_model\_path = "swin\_fusion\_model.pth"
if not os.path.exists(swin\_model\_path):
file\_id = "1cOfU1mvbGNpt0gx2hGRzseoQMJXv7F6q"
url = f"[https://drive.google.com/uc?id={file\_id}](https://drive.google.com/uc?id={file_id})"
gdown.download(url, swin\_model\_path, quiet=False)

# ✅ Load models

swin\_model = SwinClassifier()
swin\_model.load\_state\_dict(torch.load(swin\_model\_path, map\_location=torch.device('cpu'), weights\_only=False))
swin\_model.eval()

resnet\_model = ResNet18Visualizer()
resnet\_model.eval()

# ✅ Utility to convert image to bytes

def image\_to\_bytes(img: Image.Image) -> bytes:
buf = BytesIO()
img.save(buf, format="PNG")
buf.seek(0)
return buf.read()

# ✅ Page 1: Upload Image

def page\_1():
st.title("Welcome to the Image Classification App!")
st.write("Upload an image for classification.")

```
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    if st.button("Predict"):
        input_tensor, raw_img_np, pil_resized = preprocess_image(uploaded_file)

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
```

# ✅ Page 2: Prediction Results

def page\_2():
st.title(f"Prediction Results: {st.session\_state.pred\_class}")

```
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

# ✅ Display uploaded image and explanations side by side
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
```

# ✅ Preprocess image function

def preprocess\_image(img\_file):
pil\_img = Image.open(img\_file).convert('RGB')
transform\_tensor = transforms.Compose(\[
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=\[0.485, 0.456, 0.406],
std=\[0.229, 0.224, 0.225])
])
tensor\_img = transform\_tensor(pil\_img).unsqueeze(0)
raw\_img\_np = np.array(pil\_img.resize((224, 224))).astype(np.float32) / 255.0
return tensor\_img, raw\_img\_np, pil\_img.resize((224, 224))

# ✅ Main function to control pages

def main():
st.set\_page\_config(page\_title="Streamlit Multi-Page App")

```
if 'page' not in st.session_state:
    st.session_state.page = 1
    st.session_state.pred_class = None

if st.session_state.page == 1:
    page_1()
elif st.session_state.page == 2:
    page_2()
```

if **name** == "**main**":
main()

