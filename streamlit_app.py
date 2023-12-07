import streamlit as st
from PIL import Image
import io
import requests
import torch
import torch.nn as nn
import torchvision.models as models

model_weight_url = 'https://github.com/VinhRP/Makeup_Detection_Yolov8/releases/download/Makeup_detection/ResNet50.pth'
response = requests.get(model_weight_url)
model_weight_data = response.content
parameters = torch.load(io.BytesIO(model_weight_data), map_location=torch.device('cpu'))  
model = None
model = models.resnet50(weights=None)
model.load_state_dict(parameters)

image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if image_file is not None:
  our_image = Image.open(image_file).convert("RGB")
st.image(our_image, channels="RGB")
