"""
HRTNet — Multi-Retinal Disease Detection
Streamlit dashboard with collapsible sidebar navigation.

Usage:
    streamlit run app/streamlit_app.py

UI-only file. All model loading, inference, thresholding, and Grad-CAM
logic is reused unmodified from src/inference/predict.py,
src/interpretability/gradcam.py, and src/data/augmentation.py.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import base64
from io import BytesIO
from datetime import datetime

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.inference.predict import load_model_for_inference, predict_paired_images, CLASSES, CLASS_NAMES

st.set_page_config(
    page_title="HRTNet — Retinal Disease Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

FULL_NAMES = {
    "N": "Normal", "D": "Diabetes", "G": "Glaucoma", "C": "Cataract",
    "A": "Age-related Macular Degeneration", "H": "Hypertension",
    "M": "Myopia", "O": "Other",
}

PER_CLASS_METRICS = {
    "N": {"precision": 0.4581, "recall": 0.8272, "f1": 0.5897},
    "D": {"precision": 0.5152, "recall": 0.7212, "f1": 0.6010},
    "G": {"precision": 0.5682, "recall": 0.4032, "f1": 0.4717},
    "C": {"precision": 0.8000, "recall": 0.6984, "f1": 0.7458},
    "A": {"precision": 0.6486, "recall": 0.5000, "f1": 0.5647},
    "H": {"precision": 0.2414, "recall": 0.2258, "f1": 0.2333},
    "M": {"precision": 0.9744, "recall": 0.7308, "f1": 0.8352},
    "O": {"precision": 0.3518, "recall": 0.6089, "f1": 0.4459},
}

NAV_PAGES = [
    ("🏠", "Dashboard"),
    ("🕐", "Analysis History"),
    ("📊", "Model Performance"),
    ("🔬", "Research & Experiments"),
    ("ℹ️", "About HRTNet"),
]

ACCENT      = "#1AA6A0"
ACCENT_DARK = "#0E6E6A"
ACCENT_LIGHT= "#5ED8D2"
WARN        = "#D97706"

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "active_page" not in st.session_state:
    st.session_state.active_page = "Dashboard"

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@500;600&display=swap');

/* ── Reset & base ── */
html, body, .stApp {
    background: #EEF2F7;
    font-family: 'Inter', sans-serif;
    color: #16232B;
}

/* ── Hide ALL Streamlit chrome: header, toolbar, decoration, status, footer, menu ── */
header[data-testid="stHeader"]   { display: none !important; }
[data-testid="stToolbar"]        { display: none !important; }
[data-testid="stDecoration"]     { display: none !important; }
[data-testid="stStatusWidget"]   { display: none !important; }
footer                           { display: none !important; }
#MainMenu                        { display: none !important; }

/* ── Main content area ── */
.block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: none;
}

/* ── Sidebar shell ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1929 0%, #0F2236 60%, #0B1929 100%) !important;
    border-right: 1px solid #1E3148 !important;
    min-width: 260px !important;
    max-width: 260px !important;
}
section[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}

/* ── Sidebar collapse toggle (arrow button Streamlit renders natively) ── */
[data-testid="collapsedControl"] {
    background: #0E6E6A !important;
    border-radius: 0 8px 8px 0 !important;
    color: #FFFFFF !important;
    width: 26px !important;
    top: 1.2rem !important;
}
[data-testid="collapsedControl"] svg { color: #FFFFFF !important; fill: #FFFFFF !important; }

/* ── Sidebar nav buttons ── */
section[data-testid="stSidebar"] button,
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    color: #C9D6E3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    padding: 0.62rem 1rem !important;
    text-align: left !important;
    width: 100% !important;
    transition: background 0.18s ease, color 0.18s ease !important;
    margin-bottom: 0.1rem !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] button:hover,
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(26,166,160,0.15) !important;
    color: #5ED8D2 !important;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(120deg, #0E6E6A 0%, #1AA6A0 55%, #2E86D9 100%);
    border-radius: 16px;
    padding: 1.9rem 2.2rem;
    color: #FFFFFF;
    margin-bottom: 1.4rem;
    box-shadow: 0 6px 24px rgba(14,110,106,0.28);
}
.hero-title { font-size: 1.9rem; font-weight: 800; margin: 0; letter-spacing: -0.01em; }
.hero-sub   { font-size: 1rem; opacity: 0.92; margin-top: 0.3rem; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.38);
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.06em; text-transform: uppercase;
    padding: 0.28rem 0.82rem; border-radius: 20px; margin-top: 0.7rem;
}

/* ── Cards ── */
.card {
    background: #FFFFFF;
    border: 1px solid #DDE5EC;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 1px 6px rgba(16,40,48,0.06);
    height: 100%;
}
.card-title { font-size: 1.0rem; font-weight: 700; margin-bottom: 0.2rem; color: #16232B; }
.card-sub   { font-size: 0.85rem; color: #64748B; margin-bottom: 0.9rem; }

.stat-card  { text-align: left; }
.stat-label { font-size: 0.75rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.stat-value { font-size: 1.75rem; font-weight: 800; color: #0E6E6A; font-family: 'IBM Plex Mono', monospace; margin-top: 0.15rem; }

/* ── Section labels ── */
.section-label {
    font-size: 0.76rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
    color: #0E6E6A; margin: 1.6rem 0 0.7rem 0;
}

/* ── Prediction UI ── */
.top-pred-name { font-size: 1.65rem; font-weight: 800; color: #16232B; margin: 0.1rem 0; }
.top-pred-tag  { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: #0E6E6A; }

.pred-row { margin-bottom: 0.8rem; }
.pred-row-top   { display: flex; justify-content: space-between; margin-bottom: 0.25rem; font-size: 0.92rem; }
.pred-row-name  { font-weight: 500; }
.pred-row-flag  { color: #D97706; font-size: 0.68rem; font-weight: 700; margin-left: 0.4rem; text-transform: uppercase; }
.pred-row-value { font-family: 'IBM Plex Mono', monospace; color: #64748B; font-size: 0.86rem; }
.pred-track     { height: 8px; background: #EEF2F7; border-radius: 4px; overflow: hidden; }
.pred-fill         { height: 100%; border-radius: 4px; background: #1AA6A0; }
.pred-fill-flagged { background: #D97706; }

/* ── Disclaimer ── */
.disclaimer {
    background: #FFF7ED; border: 1px solid #FBD7A5; border-radius: 10px;
    padding: 0.9rem 1.2rem; font-size: 0.86rem; color: #92400E; margin-top: 1.4rem;
}

/* ── Image frame ── */
.img-frame { border-radius: 12px; overflow: hidden; border: 1px solid #DDE5EC; background: #EEF2F7; }
.img-frame img { width: 100%; display: block; }
.img-caption { font-size: 0.8rem; color: #64748B; margin-top: 0.35rem; text-align: center; }

/* ── History cards ── */
.history-card {
    background: #FFFFFF; border: 1px solid #DDE5EC; border-radius: 12px;
    padding: 0.9rem 1.1rem; margin-bottom: 0.7rem;
    display: flex; justify-content: space-between; align-items: center;
}
.history-time { font-size: 0.76rem; color: #94A3B8; font-family: 'IBM Plex Mono', monospace; }
.history-top  { font-weight: 700; font-size: 1rem; margin-top: 0.1rem; }
.history-conf { font-family: 'IBM Plex Mono', monospace; color: #0E6E6A; font-weight: 600; }

/* ── Primary action button (Analyze) — scoped to main area only ── */
div[data-testid="stMainBlockContainer"] .stButton > button {
    background: #1AA6A0 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.8rem !important;
    transition: background 0.18s ease !important;
    box-shadow: 0 2px 8px rgba(26,166,160,0.3) !important;
}
div[data-testid="stMainBlockContainer"] .stButton > button:hover {
    background: #0E6E6A !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: #F8FAFC !important;
    border: 1.5px dashed #CBD5E1 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] label p,
[data-testid="stFileUploader"] label {
    color: #16232B !important; font-weight: 600 !important; font-size: 0.92rem !important;
}
[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small { color: #475569 !important; }
[data-testid="stFileUploaderDropzone"] button {
    background: #1AA6A0 !important; color: #FFFFFF !important;
    border: none !important; border-radius: 6px !important; font-weight: 600 !important;
}
[data-testid="stFileUploaderDropzone"] button:hover { background: #0E6E6A !important; }
[data-testid="stFileUploaderDropzone"] button p    { color: #FFFFFF !important; }

hr { border-color: #DDE5EC !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_model_for_inference()


try:
    model, cfg, device = get_model()
    model_load_error = None
except Exception:
    model, cfg, device = None, None, None
    model_load_error = "The model checkpoint could not be loaded. Please confirm `checkpoints/best_model.pth` exists."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def img_tag(im):
    buf = BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<div class="img-frame"><img src="data:image/png;base64,{b64}"></div>'


def donut_gauge(pct, label, color=ACCENT):
    fig, ax = plt.subplots(figsize=(2.6, 2.6), subplot_kw={"aspect": "equal"})
    fig.patch.set_alpha(0)
    ax.pie(
        [pct, 100 - pct],
        colors=[color, "#EAEFF3"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.28, "edgecolor": "none"},
    )
    ax.text(0, 0.12, f"{pct:.0f}%", ha="center", va="center", fontsize=22, fontweight="bold", color="#16232B")
    ax.text(0, -0.22, label, ha="center", va="center", fontsize=9, color="#64748B")
    buf = BytesIO()
    plt.savefig(buf, format="png", transparent=True, bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


def full_name(code_or_display_name):
    if code_or_display_name in FULL_NAMES:
        return FULL_NAMES[code_or_display_name]
    matches = [k for k, v in CLASS_NAMES.items() if v == code_or_display_name]
    return FULL_NAMES.get(matches[0], code_or_display_name) if matches else code_or_display_name


# ---------------------------------------------------------------------------
# Sidebar — brand header + navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    # Brand header
    st.markdown("""
    <div style="
        padding: 1.6rem 1.2rem 1.2rem 1.2rem;
        border-bottom: 1px solid #1E3148;
        margin-bottom: 0.6rem;
    ">
        <div style="
            font-size: 1.2rem; font-weight: 800;
            color: #FFFFFF; letter-spacing: -0.01em; line-height: 1.2;
        ">HRT<span style="color:#5ED8D2;">Net</span></div>
        <div style="
            font-size: 0.75rem; color: #6B8BAF;
            font-weight: 500; margin-top: 0.3rem; letter-spacing: 0.02em;
        ">Retinal Disease Detection</div>
        <div style="
            display: inline-block; margin-top: 0.7rem;
            background: rgba(26,166,160,0.18); border: 1px solid rgba(94,216,210,0.35);
            font-size: 0.65rem; font-weight: 700; letter-spacing: 0.07em;
            text-transform: uppercase; padding: 0.22rem 0.65rem;
            border-radius: 20px; color: #5ED8D2;
        ">AI Research Prototype</div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation label
    st.markdown("""
    <div style="
        font-size: 0.65rem; font-weight: 700; letter-spacing: 0.10em;
        text-transform: uppercase; color: #3D5A7A;
        padding: 0 1.2rem; margin-bottom: 0.4rem;
    ">Navigation</div>
    """, unsafe_allow_html=True)

    # Nav buttons
    for icon, nav_page in NAV_PAGES:
        is_active = st.session_state.active_page == nav_page
        if is_active:
            st.markdown(f"""
            <style>
            div[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] button[kind="secondary"]:has(p:contains("{icon}")) {{
                background: rgba(26,166,160,0.25) !important;
                color: #5ED8D2 !important;
                font-weight: 700 !important;
            }}
            </style>
            """, unsafe_allow_html=True)
        if st.button(f"{icon}  {nav_page}", key=f"nav_{nav_page}", use_container_width=True):
            st.session_state.active_page = nav_page
            st.rerun()

    # Sidebar footer
    st.markdown("""
    <div style="
        margin-top: 2rem;
        padding: 0.9rem 1.2rem 0 1.2rem;
        border-top: 1px solid #1E3148;
    ">
        <div style="font-size: 0.72rem; color: #3D5A7A; line-height: 1.6;">
            ODIR-5K Dataset<br>
            <span style="color: #2A4A6A;">Epoch 68 &middot; Macro-F1 56.91%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Guard — model load error shown after sidebar renders
# ---------------------------------------------------------------------------
if model_load_error:
    st.error(model_load_error)
    st.stop()

# Route to page
page = st.session_state.active_page


# ===========================================================================
# PAGE: Dashboard
# ===========================================================================
if page == "Dashboard":
    st.markdown("""
    <div class="hero">
        <p class="hero-title">HRTNet — Multi-Retinal Disease Detection</p>
        <p class="hero-sub">Hybrid CNN + Transformer based retinal disease analysis</p>
        <span class="hero-badge">AI Research Prototype</span>
    </div>
    """, unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    guide_steps = [
        ("1", "Upload", "Provide both left and right eye fundus photographs below."),
        ("2", "Analyze", "HRTNet extracts features and generates predictions across 8 conditions."),
        ("3", "Review", "See ranked predictions, confidence, and a Grad-CAM explanation."),
    ]
    for col, (num, title, desc) in zip([g1, g2, g3], guide_steps):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:left;">
                <div style="width:32px;height:32px;border-radius:50%;background:#E4F3F4;color:#0E6E6A;
                            display:flex;align-items:center;justify-content:center;font-weight:800;margin-bottom:0.6rem;">{num}</div>
                <div style="font-weight:700;font-size:0.98rem;margin-bottom:0.25rem;">{title}</div>
                <div style="font-size:0.85rem;color:#64748B;line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Upload Retinal Fundus Images</div>', unsafe_allow_html=True)
    st.markdown('<div class="card" style="margin-bottom:1.1rem;"><div class="card-sub" style="margin-bottom:0;">Upload the left and right eye fundus photographs to analyze potential retinal disease patterns.</div></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        left_file = st.file_uploader("Left eye fundus image", type=["jpg", "jpeg", "png"])
    with col2:
        right_file = st.file_uploader("Right eye fundus image", type=["jpg", "jpeg", "png"])

    if left_file and right_file:
        try:
            left_img = Image.open(left_file).convert("RGB")
            right_img = Image.open(right_file).convert("RGB")
        except Exception:
            st.error("One of the uploaded files could not be read as an image. Please upload valid JPG or PNG fundus photographs.")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(img_tag(left_img) + '<div class="img-caption">Left eye</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(img_tag(right_img) + '<div class="img-caption">Right eye</div>', unsafe_allow_html=True)

        left_path, right_path = "/tmp/left_eye.jpg", "/tmp/right_eye.jpg"
        left_img.save(left_path)
        right_img.save(right_path)

        st.write("")
        analyze = st.button("Analyze Image", type="primary")

        if analyze:
            status = st.empty()
            try:
                status.info("Analyzing retinal image...")
                predictions = predict_paired_images(model, cfg, device, left_path, right_path)
                status.info("Extracting visual features...")
                status.info("Generating predictions...")
                status.empty()
            except Exception:
                status.empty()
                st.error("Something went wrong while analyzing the image. Please try again with different fundus photographs.")
                st.stop()

            sorted_preds = sorted(predictions.items(), key=lambda x: x[1]["probability"], reverse=True)
            top_class, top_result = sorted_preds[0]
            top_full = full_name(top_class)

            st.session_state.history.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "top_class": top_full,
                "confidence": top_result["probability"],
            })

            st.markdown('<div class="section-label">Prediction Results</div>', unsafe_allow_html=True)
            rcol1, rcol2 = st.columns([1, 1.4])

            with rcol1:
                gauge_b64 = donut_gauge(top_result["probability"] * 100, "confidence",
                                         color=WARN if top_result["positive"] else ACCENT)
                st.markdown(f"""
                <div class="card" style="text-align:center;">
                    <div class="top-pred-tag">Top Predicted Condition</div>
                    <div class="top-pred-name">{top_full}</div>
                    <img src="data:image/png;base64,{gauge_b64}" style="margin-top:0.4rem;">
                </div>
                """, unsafe_allow_html=True)

            with rcol2:
                rows = []
                for class_name, result in sorted_preds:
                    fname = full_name(class_name)
                    prob = result["probability"]
                    flagged = result["positive"]
                    fill_class = "pred-fill-flagged" if flagged else ""
                    flag_tag = '<span class="pred-row-flag">Flagged</span>' if flagged else ""
                    rows.append(
                        f'<div class="pred-row"><div class="pred-row-top">'
                        f'<span class="pred-row-name">{fname}{flag_tag}</span>'
                        f'<span class="pred-row-value">{prob:.1%}</span></div>'
                        f'<div class="pred-track"><div class="pred-fill {fill_class}" style="width:{prob*100:.1f}%"></div></div></div>'
                    )
                st.markdown(f'<div class="card"><div class="card-title">All Predicted Conditions</div>{"".join(rows)}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-label">Why did the model predict this?</div>', unsafe_allow_html=True)
            gradcam_ok, vis = True, None
            try:
                from src.interpretability.gradcam import RetinalGradCAM
                from src.data.augmentation import get_eval_transforms

                target_layer = model.backbone.backbone.blocks[-1]
                grad_cam = RetinalGradCAM(model, target_layer)
                eval_transform = get_eval_transforms(cfg["data"]["image_size"], cfg["augmentation"]["normalize_mean"], cfg["augmentation"]["normalize_std"])
                left_arr = np.array(left_img.resize((cfg["data"]["image_size"], cfg["data"]["image_size"])))
                rgb_float = left_arr.astype(np.float32) / 255.0
                tensor = eval_transform(image=left_arr)["image"].unsqueeze(0).to(device)
                code_lookup = {v: k for k, v in CLASS_NAMES.items()}
                class_idx = CLASSES.index(code_lookup[top_class])
                vis = grad_cam.generate(tensor, class_idx, rgb_float)
            except Exception:
                gradcam_ok = False

            if gradcam_ok and vis is not None:
                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    st.markdown(img_tag(left_img.resize((cfg["data"]["image_size"], cfg["data"]["image_size"]))) + '<div class="img-caption">Original (left eye)</div>', unsafe_allow_html=True)
                with gcol2:
                    buf = BytesIO()
                    Image.fromarray(vis).save(buf, format="PNG")
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    st.markdown(f'<div class="img-frame"><img src="data:image/png;base64,{b64}"></div><div class="img-caption">Grad-CAM overlay — {top_full}</div>', unsafe_allow_html=True)
                st.caption("Highlighted regions represent areas that contributed more strongly to the model's prediction. This does not necessarily indicate the precise location of disease.")
            else:
                st.info("Grad-CAM visualization could not be generated for this image. Prediction results above are unaffected.")

            st.markdown('<div class="disclaimer">This is a college project demo, not a certified diagnostic tool. Predictions should not be used for real clinical decisions.</div>', unsafe_allow_html=True)
    else:
        st.info("Upload both left and right eye fundus images above, then select **Analyze Image** to begin.")


# ===========================================================================
# PAGE: Analysis History
# ===========================================================================
elif page == "Analysis History":
    st.markdown("""
    <div class="hero">
        <p class="hero-title">Analysis History</p>
        <p class="hero-sub">A record of analyses run during this session</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No analyses yet. Run a prediction from the Dashboard to see it appear here.")
    else:
        for entry in reversed(st.session_state.history):
            st.markdown(f"""
            <div class="history-card">
                <div>
                    <div class="history-time">{entry['time']}</div>
                    <div class="history-top">{entry['top_class']}</div>
                </div>
                <div class="history-conf">{entry['confidence']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        st.caption("History is kept for the current browser session only and resets when the app restarts.")


# ===========================================================================
# PAGE: Model Performance
# ===========================================================================
elif page == "Model Performance":
    st.markdown("""
    <div class="hero">
        <p class="hero-title">Model Performance</p>
        <p class="hero-sub">Validation and test-set results for the selected checkpoint</p>
    </div>
    """, unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    for col, label, value in zip(
        [p1, p2, p3],
        ["Validation Macro-F1", "Tuned Test Macro-F1", "Best Epoch"],
        ["56.91%", "56.09%", "68"],
    ):
        with col:
            st.markdown(f'<div class="card stat-card"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Per-Class Test Performance (Tuned Thresholds)</div>', unsafe_allow_html=True)
    rows = []
    for code, m in PER_CLASS_METRICS.items():
        rows.append(
            f'<div class="pred-row"><div class="pred-row-top">'
            f'<span class="pred-row-name">{FULL_NAMES[code]}</span>'
            f'<span class="pred-row-value">F1 {m["f1"]:.2f} &nbsp;&middot;&nbsp; P {m["precision"]:.2f} &nbsp;&middot;&nbsp; R {m["recall"]:.2f}</span></div>'
            f'<div class="pred-track"><div class="pred-fill" style="width:{m["f1"]*100:.1f}%"></div></div></div>'
        )
    st.markdown(f'<div class="card">{"".join(rows)}</div>', unsafe_allow_html=True)
    st.caption("Macro-F1 is reported rather than accuracy, since this is an imbalanced multi-label classification task.")


# ===========================================================================
# PAGE: Research & Experiments
# ===========================================================================
elif page == "Research & Experiments":
    st.markdown("""
    <div class="hero">
        <p class="hero-title">Research & Experiments</p>
        <p class="hero-sub">How the final model configuration was selected</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">Selection Methodology</div>
        <div class="card-sub" style="margin-bottom:0.8rem;">
        Multiple training configurations were evaluated using validation macro-F1 as the
        selection criterion, following a controlled, one-change-at-a-time approach so each
        result could be attributed to a specific design decision.
        </div>
        <ul style="color:#334155; font-size:0.92rem; line-height:1.7; margin:0; padding-left:1.2rem;">
            <li><b>Backbone warmup:</b> the pretrained EfficientNet-B4 and Swin-B backbones
            were kept frozen for the first 5 training epochs, allowing the newly-initialized
            CBAM, FPN, and classification head layers to stabilize before fine-tuning the
            full network end-to-end.</li>
            <li><b>Class-imbalance handling:</b> a weighted sampler and an asymmetric loss
            function were used together to address ODIR-5K's class imbalance, particularly
            for minority classes such as Hypertension.</li>
            <li><b>Per-class threshold tuning:</b> rather than a single fixed decision
            threshold, each class's threshold was tuned independently on the validation
            set, which meaningfully improved test-set macro-F1.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Final Selected Configuration</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <table style="width:100%; font-size:0.92rem; color:#334155; border-collapse:collapse;">
            <tr><td style="padding:0.4rem 0; color:#64748B;">Backbone</td><td style="padding:0.4rem 0; font-weight:600;">EfficientNet-B4 (ImageNet pretrained)</td></tr>
            <tr><td style="padding:0.4rem 0; color:#64748B;">Attention</td><td style="padding:0.4rem 0; font-weight:600;">CBAM (channel + spatial)</td></tr>
            <tr><td style="padding:0.4rem 0; color:#64748B;">Neck</td><td style="padding:0.4rem 0; font-weight:600;">FPN, 4-level multi-scale fusion</td></tr>
            <tr><td style="padding:0.4rem 0; color:#64748B;">Transformer</td><td style="padding:0.4rem 0; font-weight:600;">Swin-B (ImageNet pretrained)</td></tr>
            <tr><td style="padding:0.4rem 0; color:#64748B;">Loss</td><td style="padding:0.4rem 0; font-weight:600;">Asymmetric Loss (gamma_neg=5, gamma_pos=1)</td></tr>
            <tr><td style="padding:0.4rem 0; color:#64748B;">Sampling</td><td style="padding:0.4rem 0; font-weight:600;">Inverse-frequency weighted sampler</td></tr>
            <tr><td style="padding:0.4rem 0; color:#64748B;">Best checkpoint</td><td style="padding:0.4rem 0; font-weight:600;">Epoch 68, Validation Macro-F1 0.5691</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# PAGE: About HRTNet
# ===========================================================================
elif page == "About HRTNet":
    st.markdown("""
    <div class="hero">
        <p class="hero-title">About HRTNet</p>
        <p class="hero-sub">Hybrid CNN + Transformer for retinal disease detection</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">Architecture</div>
            <div class="card-sub" style="margin-bottom:0.6rem;">A hybrid CNN-Transformer pipeline combining local feature extraction with global attention.</div>
            <p style="font-size:0.9rem; color:#334155; line-height:1.7; margin:0;">
            Fundus image &rarr; <b>EfficientNet-B4</b> backbone &rarr; <b>CBAM</b> attention
            (per stage) &rarr; <b>FPN</b> multi-scale feature fusion &rarr; fusion bridge &rarr;
            <b>Swin-B</b> Transformer encoder &rarr; classification head &rarr; 8 independent
            disease probabilities.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Dataset</div>
            <div class="card-sub" style="margin-bottom:0.6rem;">ODIR-5K — Ocular Disease Intelligent Recognition</div>
            <p style="font-size:0.9rem; color:#334155; line-height:1.7; margin:0;">
            Fundus photographs from 5,000 patients, labeled across 8 disease categories
            by trained ophthalmologists. Each eye is treated as an individual training
            sample carrying its patient's diagnosis.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Disease Classes</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (code, name) in enumerate(FULL_NAMES.items()):
        with cols[i % 4]:
            st.markdown(f'<div class="card" style="text-align:center; padding:1rem;"><div style="font-weight:700;">{name}</div><div style="color:#94A3B8; font-family:\'IBM Plex Mono\',monospace; font-size:0.8rem;">{code}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">How Predictions Work</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p style="font-size:0.9rem; color:#334155; line-height:1.7; margin:0;">
        HRTNet accepts left and right eye fundus images together. Each eye is processed
        through the shared network, and the resulting features are combined before the
        final prediction &mdash; reflecting the fact that these conditions are typically
        assessed by examining both eyes. Predictions are independent for each of the 8
        classes, since a patient may present with more than one condition at once.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="disclaimer">This is a college project demo, not a certified diagnostic tool. Predictions should not be used for real clinical decisions.</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#94A3B8; font-size:0.8rem; margin-top:2.5rem; padding-top:1.2rem; border-top:1px solid #DDE5EC;">
    HRTNet &bull; Retinal Disease Detection &nbsp;|&nbsp; College Final-Year Project &bull; AI Research Prototype
</div>
""", unsafe_allow_html=True)