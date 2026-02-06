from unittest import result
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
import pandas as pd

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Object Detection App",
    layout="wide"
)

# ---------------------------
# Custom CSS for UI
# ---------------------------
st.markdown("""
<style>
.stApp { background-color: #f5f5f5; font-family: 'Segoe UI', sans-serif; }
.card { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); margin-bottom: 20px; }
h1 { color: #4B0082; text-align: center; }
.stButton>button { background-color: #4B0082; color: white; border-radius: 8px; padding: 8px 18px; font-weight: bold; transition: all 0.3s ease; }
.stButton>button:hover { background-color: #6a0dad; color: white; }
.sidebar .sidebar-content h2 { color: #4B0082; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# App Header
# ---------------------------
st.markdown("<h1>Object Detection App</h1>", unsafe_allow_html=True)
st.markdown('<p style="text-align:center;">Detect multiple objects in images or live from webcam in real-time!</p>', unsafe_allow_html=True)

# ---------------------------
# Load YOLOv8 model
# ---------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# ---------------------------
# Choose Mode
# ---------------------------
mode = st.radio(
    "Choose Detection Mode:",
    ["📸 Single Image", "📂 Multiple Images", "🎥 Live Webcam"],
    horizontal=True
)

# =============================
# FUNCTION TO DISPLAY RESULTS SIDE-BY-SIDE WITH TABLE
# =============================
def display_side_by_side_table(original_image, results, uid):
    # Annotated image
    res_plotted = results[0].plot()
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'><b>Original Image</b></div>", unsafe_allow_html=True)
        st.image(original_image, use_container_width=True)

    with col2:
        st.markdown("<div class='card'><b>Detection Result</b></div>", unsafe_allow_html=True)
        st.image(res_rgb, use_container_width=True)

        _, output_path = tempfile.mkstemp(suffix=".jpg")
        cv2.imwrite(output_path, cv2.cvtColor(res_rgb, cv2.COLOR_RGB2BGR))

        with open(output_path, "rb") as file:
            st.download_button(
                label="💾 Download Result",
                data=file,
                file_name="detected_image.jpg",
                key=f"download_result_{uid}"
            )

    # Detected objects table
    if len(results[0].boxes) > 0:
        detected_classes = [model.names[int(c)] for c in results[0].boxes.cls.cpu().numpy()]
        boxes = results[0].boxes.xyxy.cpu().numpy()

        df = pd.DataFrame({
            "Class": detected_classes,
            "x1": boxes[:, 0].astype(int),
            "y1": boxes[:, 1].astype(int),
            "x2": boxes[:, 2].astype(int),
            "y2": boxes[:, 3].astype(int)
        })

        df_count = df["Class"].value_counts().reset_index()
        df_count.columns = ["Class", "Count"]

        st.markdown("<div class='card'><b>Detected Objects Summary</b></div>", unsafe_allow_html=True)
        st.table(df_count)
    else:
        st.warning("No objects detected!")

# =============================
# 📸 SINGLE IMAGE MODE
# =============================
if mode == "📸 Single Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        tfile_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(tfile_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        original_image = Image.open(uploaded_file)
        with st.spinner("Detecting objects..."):
            results = model.predict(source=tfile_path, conf=0.25)

        display_side_by_side_table(original_image, results, uid="single")
        try:
            os.remove(tfile_path)
        except:
            pass

# =============================
# 📂 MULTIPLE IMAGE MODE
# =============================
elif mode == "📂 Multiple Images":
    uploaded_files = st.file_uploader("Upload Multiple Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        st.info(f"{len(uploaded_files)} image(s) uploaded. Click below to start batch detection.")
        if st.button("🚀 Run Detection on All Images"):
            for idx, uploaded_file in enumerate(uploaded_files):
                tfile_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                with open(tfile_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                original_image = Image.open(uploaded_file)
                with st.spinner(f"Detecting objects in {uploaded_file.name}..."):
                    results = model.predict(source=tfile_path, conf=0.25)

                st.subheader(f"Result - {uploaded_file.name}")
                display_side_by_side_table(original_image, results, uid=idx)

                try:
                    os.remove(tfile_path)
                except:
                    pass
    else:
        st.info("📂 Upload multiple images to start detection.")

# =============================
# 🎥 LIVE WEBCAM DETECTION MODE WITH LIVE SUMMARY
# =============================
else:
    start_button = st.button("Start Webcam Detection")
    stop_button = st.button("Stop Webcam Detection")
    stframe = st.empty()

    if start_button:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Unable to access the webcam.")
        else:
            st.success("✅ Webcam started. Press 'Stop' to end.")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("⚠️ Failed to grab frame from webcam.")
                    break

                results = model.predict(source=frame, conf=0.5, verbose=False)
                annotated_frame = results[0].plot()
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_container_width=True)

                # Sidebar live summary
                if len(results[0].boxes) > 0:
                    detected_classes = [model.names[int(c)] for c in results[0].boxes.cls.cpu().numpy()]
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    df = pd.DataFrame({
                        "Class": detected_classes,
                        "x1": boxes[:,0].astype(int),
                        "y1": boxes[:,1].astype(int),
                        "x2": boxes[:,2].astype(int),
                        "y2": boxes[:,3].astype(int)
                    })
                    df_count = df["Class"].value_counts().reset_index()
                    df_count.columns = ["Class", "Count"]
                    st.sidebar.markdown("### Live Object Summary")
                    st.sidebar.table(df_count)

                time.sleep(0.03)
                if stop_button:
                    break

            cap.release()
            st.info("🛑 Webcam stopped.")
