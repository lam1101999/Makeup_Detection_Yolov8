import streamlit as st
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
import torch
import torchvision
from torchvision import transforms
from typing import List, Tuple
import ultralytics
ultralytics.checks()
from ultralytics import YOLO

image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if image_file is not None:
  our_image = Image.open(image_file).convert("RGB")
  yolo = YOLO('./face_detection_224.pt')
  results = yolo.predict(our_image) # predict on an image
