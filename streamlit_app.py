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
res50 = None
res50 = models.resnet50(weights=None)
numFeatures = res50.fc.in_features

headModel = torch.nn.Sequential(
	torch.nn.Linear(numFeatures, 512),
	torch.nn.ReLU(),
	torch.nn.Dropout(0.25),
	torch.nn.Linear(512, 256),
	torch.nn.ReLU(),
	torch.nn.Dropout(0.5),
	torch.nn.Linear(256, 2)
)
res50.fc = headModel
res50.load_state_dict(parameters)

image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if image_file is not None:
  our_image = Image.open(image_file).convert("RGB")
st.image(our_image, channels="RGB")
