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

from src.inference.predict import load_model_for_inference, predict_paired_images, CLASSES, CLASS_NAMES

st.set_page_config(
    page_title="Retinal Disease Detection",
    page_icon="\u25c9",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Visual identity: deep clinical dark theme with an ophthalmoscope motif.
# Palette: near-black teal (fundus-photo darkness), warm amber (optic-disc
# glow), clinical red (flagged findings), warm off-white (report text).
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #0D1512;
    --surface: #16211C;
    --surface-raised: #1C2921;
    --border: #2A3830;
    --accent: #E8A94E;
    --accent-dim: #8A6A38;
    --alert: #C4453A;
    --alert-dim: #6B3430;
    --text: #F0EDE4;
    --text-muted: #8A9990;
    --text-faint: #556258;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

/* Hide default Streamlit chrome */
#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; height: 0; }
.block-container { padding-top: 2.5rem; max-width: 1100px; }

/* Typography */
h1, h2, h3, .report-heading {
    font-family: 'Newsreader', serif !important;
    color: var(--text) !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em;
}
body, p, div, span, label {
    font-family: 'Inter', sans-serif;
}
.mono, .prob-value, .threshold-tag {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Header block */
.scope-header {
    display: flex;
    align-items: baseline;
    gap: 0.9rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.1rem;
    margin-bottom: 0.3rem;
}
.scope-mark {
    font-size: 1.6rem;
    color: var(--accent);
    line-height: 1;
}
.scope-title {
    font-family: 'Newsreader', serif;
    font-size: 1.85rem;
    font-weight: 500;
    color: var(--text);
    margin: 0;
}
.scope-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: var(--text-muted);
    letter-spacing: 0.03em;
    text-transform: uppercase;
    margin-top: 0.35rem;
}

/* Ophthalmoscope-style circular image frame */
.scope-frame {
    position: relative;
    width: 100%;
    aspect-ratio: 1 / 1;
    border-radius: 50%;
    overflow: hidden;
    border: 1px solid var(--border);
    box-shadow: inset 0 0 40px rgba(0,0,0,0.65), 0 0 0 6px var(--bg), 0 0 0 7px var(--border);
    background: #060907;
}
.scope-frame img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.scope-label {
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-top: 0.7rem;
}

/* Section eyebrow */
.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.6rem;
    margin-top: 2.2rem;
}

/* Readout rows (predictions list) */
.readout-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.65rem 0;
    border-bottom: 1px solid var(--border);
}
.readout-row:last-child { border-bottom: none; }
.readout-left {
    display: flex;
    align-items: center;
    gap: 0.7rem;
}
.readout-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-flagged { background: var(--alert); box-shadow: 0 0 8px rgba(196,69,58,0.6); }
.dot-clear { background: var(--text-faint); }
.readout-name {
    font-size: 0.95rem;
    color: var(--text);
}
.readout-right {
    display: flex;
    align-items: center;
    gap: 0.9rem;
}
.prob-track {
    width: 120px;
    height: 4px;
    background: var(--surface-raised);
    border-radius: 2px;
    overflow: hidden;
}
.prob-fill { height: 100%; border-radius: 2px; }
.fill-flagged { background: var(--alert); }
.fill-clear { background: var(--text-faint); }
.prob-value {
    font-size: 0.85rem;
    color: var(--text);
    min-width: 3.4rem;
    text-align: right;
}

/* Disclaimer */
.clinical-note {
    margin-top: 2.5rem;
    padding: 0.9rem 1.1rem;
    background: var(--surface);
    border-left: 2px solid var(--accent-dim);
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.5;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #16130B !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    padding: 0.55rem 1.6rem !important;
}
.stButton > button:hover { background: #F2BC6E !important; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
}

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_model_for_inference()


model, cfg, device = get_model()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="scope-header">
    <span class="scope-mark">&#9678;</span>
    <div>
        <p class="scope-title">Retinal Disease Detection</p>
        <p class="scope-subtitle">EfficientNet-B4 &middot; CBAM &middot; FPN &middot; Swin-B &nbsp;/&nbsp; trained on ODIR-5K</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Fundus Photographs</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    left_file = st.file_uploader("Left eye", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.caption("Left eye")
with col2:
    right_file = st.file_uploader("Right eye", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.caption("Right eye")

if left_file and right_file:
    left_img = Image.open(left_file).convert("RGB")
    right_img = Image.open(right_file).convert("RGB")

    import base64
    from io import BytesIO

    def _img_to_b64(im):
        buf = BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="scope-frame"><img src="data:image/png;base64,{_img_to_b64(left_img)}"></div>'
            f'<div class="scope-label">OS &middot; Left Eye</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="scope-frame"><img src="data:image/png;base64,{_img_to_b64(right_img)}"></div>'
            f'<div class="scope-label">OD &middot; Right Eye</div>',
            unsafe_allow_html=True,
        )

    left_path = "/tmp/left_eye.jpg"
    right_path = "/tmp/right_eye.jpg"
    left_img.save(left_path)
    right_img.save(right_path)

    st.write("")
    run = st.button("Run Prediction")

    if run:
        with st.spinner("Analyzing fundus images..."):
            predictions = predict_paired_images(model, cfg, device, left_path, right_path)

        st.markdown('<div class="eyebrow">Prediction Readout</div>', unsafe_allow_html=True)

        sorted_preds = sorted(predictions.items(), key=lambda x: x[1]["probability"], reverse=True)
        top_class, top_result = sorted_preds[0]

        rows_html = ""
        for class_name, result in sorted_preds:
            prob = result["probability"]
            flagged = result["positive"]
            dot_class = "dot-flagged" if flagged else "dot-clear"
            fill_class = "fill-flagged" if flagged else "fill-clear"
            rows_html += f"""
            <div class="readout-row">
                <div class="readout-left">
                    <span class="readout-dot {dot_class}"></span>
                    <span class="readout-name">{class_name}</span>
                </div>
                <div class="readout-right">
                    <div class="prob-track"><div class="prob-fill {fill_class}" style="width:{prob*100:.0f}%"></div></div>
                    <span class="prob-value">{prob:.1%}</span>
                </div>
            </div>
            """
        st.markdown(f'<div>{rows_html}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="eyebrow">Grad-CAM &middot; {top_class}</div>', unsafe_allow_html=True)
        with st.spinner("Generating attention overlay..."):
            from src.interpretability.gradcam import RetinalGradCAM
            from src.data.augmentation import get_eval_transforms

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
            st.image(vis, width=400)

        st.markdown(
            '<div class="clinical-note">This tool is a college research project demo and is not a '
            'certified diagnostic device. Predictions should not be used for real clinical decisions.</div>',
            unsafe_allow_html=True,
        )
else:
    st.write("")
    st.caption("Upload both left and right fundus images to begin.")