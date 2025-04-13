import streamlit as st
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
import numpy as np

# Initialize MTCNN
mtcnn = MTCNN(keep_all=True)

# Streamlit UI
st.title("Face Detection using MTCNN")
st.write("Upload an image and see the detected faces highlighted.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect faces
    boxes, _ = mtcnn.detect(image)

    # Draw bounding boxes
    if boxes is not None:
        image_with_boxes = image.copy()
        draw = ImageDraw.Draw(image_with_boxes)
        for box in boxes:
            draw.rectangle(box.tolist(), outline="red", width=3)

        st.image(image_with_boxes, caption="Detected Faces", use_column_width=True)
        st.success(f"{len(boxes)} face(s) detected.")
    else:
        st.warning("No faces detected.")
