import streamlit as st
from PIL import Image

image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if image_file is not None:
  our_image = Image.open(image_file).convert("RGB")
st.text(information)
st.image(our_image, channels="RGB")

