import streamlit as st
import numpy as np
import cv2
import os
import joblib
from skimage import feature
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from PIL import Image

IMG_SIZE = 224
MANUAL_IMG_SIZE = 32
MODEL_DIR = "Model"

CLASS_LABELS = [
    "aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
    "household electrical devices", "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large natural outdoor scenes", "large omnivores and herbivores",
    "medium-sized mammals", "non-insect invertebrates", "people", "reptiles", "small mammals",
    "trees", "vehicles 1", "vehicles 2"
]

@st.cache_resource
def get_resnet():
    base = ResNet50(include_top=False, weights="imagenet",
                    input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")
    model = Model(inputs=base.input, outputs=base.output)
    model.trainable = False
    return model

def preprocess_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return cv2.normalize(g, None, 0, 1, cv2.NORM_MINMAX).astype("float32")

def hog_f(x):
    return feature.hog(x, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm="L2-Hys")

def lbp_f(x, P=8, R=1):
    lbp = feature.local_binary_pattern(x, P, R, method="uniform")
    h, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P+3))
    return (h / (h.sum() + 1e-6)).astype("float32")

def hsv_f(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                     [0, 180, 0, 256, 0, 256])
    return cv2.normalize(h, None).flatten().astype("float32")

def manual_features(img):
    g = preprocess_gray(img)
    return np.hstack([hog_f(g), lbp_f(g), hsv_f(img)])

def resnet_extract(img, model):
    r = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    p = preprocess_input(r.astype(np.float32))[np.newaxis, ...]
    return model.predict(p, verbose=0)[0]

@st.cache_resource
def load_assets():
    needed = ["scaler_manual.pkl", "scaler_res.pkl", "pca.pkl", "hybrid_model.keras"]
    missing = [f for f in needed if not os.path.exists(os.path.join(MODEL_DIR, f))]

    if missing:
        st.error("Missing model files: " + ", ".join(missing))
        return None, None, None, None, None

    scaler_m = joblib.load(os.path.join(MODEL_DIR, "scaler_manual.pkl"))
    scaler_r = joblib.load(os.path.join(MODEL_DIR, "scaler_res.pkl"))
    pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
    clf = load_model(os.path.join(MODEL_DIR, "hybrid_model.keras"))
    res = get_resnet()

    return scaler_m, scaler_r, pca, clf, res

sc_m, sc_r, pca, clf, res = load_assets()

def predict(image, sc_m, sc_r, pca, clf, res):
    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    man = cv2.resize(rgb, (MANUAL_IMG_SIZE, MANUAL_IMG_SIZE))
    f_m = sc_m.transform([manual_features(man)])[0]
    f_r = sc_r.transform([resnet_extract(rgb, res)])[0]
    fused = np.hstack([0.5 * f_m, f_r])
    p = clf.predict(pca.transform([fused]), verbose=0)[0]
    return np.argmax(p), p

def main():
    st.set_page_config(page_title="Hybrid CIFAR-100 Classifier", layout="centered")

    st.markdown("""
    <div style="text-align:center; padding:20px;">
        <h1 style="color:#4CAF50; margin-bottom:0;">Hybrid CIFAR-100 Classifier</h1>
        <p style="font-size:16px; color:#555; margin-top:5px;">
            Combination of <b>HOG-LBP-HSV</b> as <b>manual features</b> with <b>ResNet50 embeddings</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    if clf is None:
        return

    st.markdown("<hr>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with col2:
            with st.spinner("Classifying.."):
                idx, probs = predict(img, sc_m, sc_r, pca, clf, res)

            label = CLASS_LABELS[idx]

            st.markdown(f"""
            <div style="
                background:#E8F8FF;
                padding:18px 20px;
                border-radius:12px;
                border-left:6px solid #1C90FF;">
                <h3 style="margin:0; color:#005BBB;">{label}</h3>
                <p style="font-size:18px; margin:5px 0 0;">
                    Confidence: <b>{probs[idx]*100:.2f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("Best 5 Predictions")
            best5 = np.argsort(probs)[::-1][:5]

            for i in best5:
                st.progress(float(probs[i]),
                            text=f"{CLASS_LABELS[i]} ({probs[i]*100:.2f}%)")

if __name__ == "__main__":
    main()
