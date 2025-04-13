import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
from facenet_pytorch import MTCNN

# Initialize MTCNN
mtcnn = MTCNN(keep_all=True)

# Load Viola-Jones Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_mtcnn(image):
    boxes, _ = mtcnn.detect(image)
    return boxes

def detect_faces_viola_jones(image):
    # Convert PIL to OpenCV format
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    boxes = []
    for (x, y, w, h) in faces:
        boxes.append([x, y, x + w, y + h])
    return boxes

def draw_boxes(image, boxes, color="red"):
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    for box in boxes:
        draw.rectangle(box, outline=color, width=3)
    return image_with_boxes

# Streamlit UI
st.title("🧠 Face Detection App")
st.write("Upload an image and select a model to detect faces.")

model_option = st.selectbox("Choose a model", ["MTCNN", "Viola-Jones"])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if model_option == "MTCNN":
        boxes = detect_faces_mtcnn(image)
    else:
        boxes = detect_faces_viola_jones(image)

    if boxes is not None and len(boxes) > 0:
        boxes = [list(map(int, box)) for box in boxes]
        result_image = draw_boxes(image, boxes, color="green" if model_option == "Viola-Jones" else "red")
        st.image(result_image, caption=f"{model_option} Detected Faces", use_column_width=True)
        st.success(f"{len(boxes)} face(s) detected using {model_option}.")
    else:
        st.warning(f"No faces detected using {model_option}.")
