import streamlit as st
import os
import time
from PIL import Image
import torch
import numpy as np

# --------------------------------
# Page setup
# --------------------------------

st.set_page_config(page_title="Crack Prediction", layout="wide")

st.markdown("""
<h3 style='margin-bottom:0;'>Data-Driven Prediction of Crack Formation in Ti-6242 Alloy</h3>
<p style='margin-top:5px;'>Select a Ti-6242 microstructure and let the AI predict where the crack will form during dwell fatigue.</p>
""", unsafe_allow_html=True)

st.divider()


# --------------------------------
# SESSION STATE
# --------------------------------

if "screen" not in st.session_state:
    st.session_state.screen = "gallery"

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# --------------------------------
# IMAGE DETECTION
# --------------------------------

IMAGE_FOLDER = "test_images"

image_files = sorted(
    [f for f in os.listdir(IMAGE_FOLDER)
     if f.lower().endswith((".png",".jpg",".jpeg"))],
    key=lambda x: int(os.path.splitext(x)[0])
)


# ====================================
# SCREEN 1 — GALLERY
# ====================================

if st.session_state.screen == "gallery":

    st.subheader("Available Microstructures")

    cols = st.columns(len(image_files))

    for i, file in enumerate(image_files):

        img_path = os.path.join(IMAGE_FOLDER, file)
        image = Image.open(img_path).convert("RGB")

        with cols[i]:

            st.image(image, use_column_width=True)

            if st.button(f"Sample {i+1}", key=f"btn{i}"):

                st.session_state.selected_image = file
                st.session_state.screen = "loading"
                st.rerun()

    st.stop()


# ====================================
# SCREEN 2 — LOADING TRANSITION
# ====================================

if st.session_state.screen == "loading":

    st.markdown(
        "<h3 style='text-align:center;'>Running AI Model...</h3>",
        unsafe_allow_html=True
    )

    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    # --------------------------------
    # MODEL INFERENCE (UNCHANGED)
    # --------------------------------

    img_path = os.path.join(IMAGE_FOLDER, st.session_state.selected_image)

    image = Image.open(img_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    output_img = (output.squeeze().permute(1,2,0).cpu().numpy() + 1) / 2
    output_img = np.clip(output_img, 0, 1)

    st.session_state.prediction = output_img

    st.session_state.screen = "result"

    st.rerun()


# ====================================
# SCREEN 3 — RESULT
# ====================================

if st.session_state.screen == "result":

    st.subheader("AI Crack Prediction")

    st.image(
        st.session_state.prediction,
        width=650
    )

    st.divider()

    if st.button("← Back to samples"):

        st.session_state.screen = "gallery"
        st.rerun()
