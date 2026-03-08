# ============================================================
# UI / UX LAYER
# ============================================================

# Session state initialization
if "screen" not in st.session_state:
    st.session_state.screen = "gallery"

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ------------------------------------------------------------
# Detect images automatically
# ------------------------------------------------------------

IMAGE_FOLDER = "test_images"

image_files = sorted(
    [f for f in os.listdir(IMAGE_FOLDER)
     if f.lower().endswith((".png",".jpg",".jpeg"))],
    key=lambda x: int(os.path.splitext(x)[0])
)


# ============================================================
# SCREEN 1 : GALLERY
# ============================================================

if st.session_state.screen == "gallery":

    st.markdown("""
    <h3>Data-Driven Prediction of Crack Formation in Ti-6242 Alloy</h3>
    <p>Select a Ti-6242 microstructure and let the AI predict where the crack will form during dwell fatigue.</p>
    """, unsafe_allow_html=True)

    st.divider()

    cols = st.columns(len(image_files))

    for i, file in enumerate(image_files):

        img_path = os.path.join(IMAGE_FOLDER, file)
        img = Image.open(img_path).convert("RGB")

        with cols[i]:

            st.image(img, use_column_width=True)

            if st.button(f"Sample {i+1}", key=f"s{i}"):

                st.session_state.selected_image = file
                st.session_state.screen = "loading"
                st.rerun()

    st.stop()


# ============================================================
# SCREEN 2 : LOADING TRANSITION
# ============================================================

if st.session_state.screen == "loading":

    st.markdown(
        "<h2 style='text-align:center;'>Running AI Model</h2>",
        unsafe_allow_html=True
    )

    progress = st.progress(0)

    diagnostics = st.empty()

    for i in range(100):

        progress.progress(i+1)

        diagnostics.markdown(
            f"""
            **Inference Diagnostics**

            Progress : `{i+1}%`  
            Device : `{device}`  
            Tensor : `1 x 3 x 512 x 512`  
            Network : `Attention U-Net Generator`
            """
        )

        time.sleep(0.01)


    # ----- model inference (your original logic) -----

    img_path = os.path.join(IMAGE_FOLDER, st.session_state.selected_image)
    image = Image.open(img_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    output_img = (output.squeeze().permute(1,2,0).cpu().numpy() + 1) / 2
    output_img = np.clip(output_img,0,1)

    st.session_state.prediction = output_img
    st.session_state.screen = "result"

    st.rerun()


# ============================================================
# SCREEN 3 : RESULT
# ============================================================

if st.session_state.screen == "result":

    st.markdown("### AI Crack Prediction")

    st.image(
        st.session_state.prediction,
        width=650
    )

    st.divider()

    if st.button("← Back to samples"):

        st.session_state.screen = "gallery"
        st.rerun()
