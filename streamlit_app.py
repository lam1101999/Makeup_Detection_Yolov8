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

def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict

res50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
res50 = resnet50(weights="DEFAULT")

for module, param in zip(res50.modules(), res50.parameters()):
	if isinstance(module, torch.nn.BatchNorm2d):
		param.requires_grad = False

# define the network head and attach it to the model
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

res50.load_state_dict(torch.load("https://github.com/VinhRP/Makeup_Detection_Yolov8/releases/download/Makeup_detection/ResNet50.pth"))

device = "cuda" if torch.cuda.is_available() else "cpu"

image_size: Tuple[int, int] = (224, 224)
class_names = ['Make_up', 'Non_Make_up']

image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
res50.to(device)
res50.eval()

image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if image_file is not None:
  our_image = Image.open(image_file).convert("RGB")
  yolo = YOLO('https://github.com/VinhRP/Makeup_Detection_Yolov8/releases/download/Makeup_detection/face_detection_224.pt')
  results = yolo.predict(our_image) # predict on an image
  for r in results:
    cords = r.boxes.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    img = Image.fromarray(r.plot()[..., :: -1]) ## Original image
    img2 = img.crop(cords) ##Croped image
  with torch.inference_mode():
    transformed_image = image_transform(img2).unsqueeze(dim=0)
    target_image_pred = res50(transformed_image.to(device))
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
  information = f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
  st.text(information)
  st.image(our_image, channels="RGB")
