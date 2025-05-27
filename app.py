import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from unet_model import UNet

@st.cache_resource
def load_model():
    model = UNet(n_classes=10)
    model.load_state_dict(torch.load("floodnet_unet.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    image = image.resize((512, 512))
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0)

def predict(image_tensor, model):
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).numpy()
    return pred

st.title("FloodNet Segmentation - U-Net")

uploaded_file = st.file_uploader("Upload an aerial flood image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_tensor = preprocess_image(image)
    model = load_model()
    prediction = predict(image_tensor, model)

    st.subheader("Segmentation Result")
    plt.figure(figsize=(6, 6))
    plt.imshow(prediction, cmap="tab10")
    st.pyplot(plt)
