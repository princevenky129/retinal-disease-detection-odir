"""
Streamlit demo: upload left+right fundus images, get an 8-class multi-label
prediction with confidence scores and a GradCAM overlay.

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
from PIL import Image

from src.inference.predict import load_model_for_inference, predict_paired_images, CLASS_NAMES

st.set_page_config(page_title="Retinal Disease Detection", layout="wide")
st.title("Multi-Retinal Disease Detection")
st.caption("EfficientNet-B4 + CBAM + FPN + Swin-B, trained on ODIR-5K")


@st.cache_resource
def get_model():
    return load_model_for_inference()


model, cfg, device = get_model()

col1, col2 = st.columns(2)
with col1:
    left_file = st.file_uploader("Left eye fundus image", type=["jpg", "jpeg", "png"])
with col2:
    right_file = st.file_uploader("Right eye fundus image", type=["jpg", "jpeg", "png"])

if left_file and right_file:
    left_img = Image.open(left_file).convert("RGB")
    right_img = Image.open(right_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(left_img, caption="Left eye", use_container_width=True)
    with col2:
        st.image(right_img, caption="Right eye", use_container_width=True)

    # Save temp files for the predict function (which reads from disk)
    left_path = "/tmp/left_eye.jpg"
    right_path = "/tmp/right_eye.jpg"
    left_img.save(left_path)
    right_img.save(right_path)

    if st.button("Run Prediction"):
        with st.spinner("Running inference..."):
            predictions = predict_paired_images(model, cfg, device, left_path, right_path)

        st.subheader("Predictions")
        top_class = None
        top_prob = -1
        for class_name, result in sorted(
            predictions.items(), key=lambda x: x[1]["probability"], reverse=True
        ):
            prob = result["probability"]
            flag = "🔴" if result["positive"] else "⚪"
            st.write(f"{flag} **{class_name}**: {prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))
            if prob > top_prob:
                top_prob, top_class = prob, class_name

        st.subheader(f"Why: GradCAM for top prediction ({top_class})")
        with st.spinner("Generating GradCAM overlay..."):
            from src.interpretability.gradcam import RetinalGradCAM
            from src.data.augmentation import get_eval_transforms
            from src.inference.predict import CLASSES, CLASS_NAMES
            import numpy as np

            target_layer = model.backbone.backbone.blocks[-1]
            grad_cam = RetinalGradCAM(model, target_layer)

            eval_transform = get_eval_transforms(
                cfg["data"]["image_size"], cfg["augmentation"]["normalize_mean"], cfg["augmentation"]["normalize_std"]
            )
            left_arr = np.array(left_img.resize((cfg["data"]["image_size"], cfg["data"]["image_size"])))
            rgb_float = left_arr.astype(np.float32) / 255.0
            tensor = eval_transform(image=left_arr)["image"].unsqueeze(0).to(device)

            code_lookup = {v: k for k, v in CLASS_NAMES.items()}
            class_idx = CLASSES.index(code_lookup[top_class])

            vis = grad_cam.generate(tensor, class_idx, rgb_float)
            st.image(vis, caption=f"GradCAM overlay: {top_class}", width=400)

        st.info(
            "This is a college project demo, NOT a certified diagnostic tool. "
            "Predictions should not be used for real clinical decisions."
        )

else:
    st.write("Upload both left and right fundus images to get a prediction.")
