import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import pandas as pd
import time

st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      header {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

MODEL_PATH = "Effiicientnetv2b2.keras"
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)
model = load_model()


st.markdown(
    """
    <h1 style="text-align:center; color:#3D5A80; font-size:40px; margin-bottom:0.1em;">
      🗑️ Garbage Classifier
    </h1>
    <p style="text-align:center; color:#6c757d; margin-top:0;">Powered by EfficientNetV2B2</p>
    """,
    unsafe_allow_html=True
)

icon_urls = {
    "Recycle": "https://img.icons8.com/color/96/000000/recycle-sign.png",
    "Trash Bin": "https://img.icons8.com/fluency/96/filled-trash.png",
    "Plastic": "https://img.icons8.com/fluency/96/plastic.png",
    "Glass": "https://img.icons8.com/fluency/96/wine-glass.png",
    "Cardboard": "https://img.icons8.com/fluency/96/cardboard-box.png",
    "Metal": "https://img.icons8.com/?size=100&id=S3oEDA5waPxP&format=png&color=000000"
}

cols = st.columns(len(icon_urls))
for col, (label, url) in zip(cols, icon_urls.items()):
    try:
        col.image(url, caption=label, width=70)
    except:
        col.markdown(f"**{label}**")

st.markdown("---")

if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "live" not in st.session_state:
    st.session_state.live = False

def preprocess(img: Image.Image):
    img = img.resize((124,124))
    arr = np.array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr,0)

def predict(img: Image.Image):
    x = preprocess(img)
    p = model.predict(x)
    idx = np.argmax(p)
    return CLASS_NAMES[idx], float(p[0][idx])

def show_result(label, conf):
    st.markdown("### 🔍 Prediction")
    st.success(f"**{label.upper()}** — {conf:.2%}")
    st.progress(min(int(conf*100),100))
    st.session_state.predictions.append({"class":label, "confidence":conf})

mode = st.radio(
    "Select Input Mode:",
    ['🖼️ Upload Image', '📸 Capture from Camera', '📹 Live Webcam'],
    index=0, horizontal=True, key="input_mode"
)

if mode == '🖼️ Upload Image':
    up = st.file_uploader("Upload an image", type=["jpg","jpeg","png"], key="u1")
    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="📷 Preview", use_container_width=False, width=300)
        lbl, cf = predict(img)
        show_result(lbl, cf)

elif mode == '📸 Capture from Camera':
    cam = st.camera_input("Capture image", key="c1")
    if cam:
        img = Image.open(cam).convert("RGB")
        st.image(img, caption="📷 Preview", use_container_width=False, width=300)
        lbl, cf = predict(img)
        show_result(lbl, cf)

elif mode == '📹 Live Webcam':
    start = st.button("🟢 Start Live", key="s1")
    stop  = st.button("🔴 Stop Live",  key="s2")
    ph_frame  = st.empty()
    ph_result = st.empty()
    ph_prog   = st.empty()

    if start: st.session_state.live = True
    if stop:  st.session_state.live = False

    if st.session_state.live:
        cap = cv2.VideoCapture(0)
        st.info("🔄 Live webcam running...")
        while st.session_state.live:
            ok, frame = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            lbl, cf = predict(img)
            ph_frame.image(rgb, channels="RGB", use_container_width=False, width=300)
            ph_result.markdown(f"### 🔍 **{lbl.upper()}** — {cf:.2%}")
            ph_prog.progress(min(int(cf*100),100))
            time.sleep(0.5)
        cap.release()
        ph_frame.empty(); ph_result.empty(); ph_prog.empty()

st.markdown("---")
if st.session_state.predictions:
    df = pd.DataFrame([
        {"class":p["class"], "confidence":f"{p['confidence']:.2%}"}
        for p in st.session_state.predictions
    ])
    st.markdown("### 📊 History")
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="predictions.csv", mime="text/csv")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center;">Made with ❤️ by Raman</div>', unsafe_allow_html=True)
