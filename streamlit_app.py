!pip install -q streamlit
import locale
locale.getpreferredencoding = lambda: "UTF-8"

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
from ultralytics import YOLO

image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if image_file is not None:
  our_image = Image.open(image_file).convert("RGB")
st.text(information)
st.image(our_image, channels="RGB")

