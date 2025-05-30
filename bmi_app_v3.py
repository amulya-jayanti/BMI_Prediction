import os
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 1) Load extractor
@st.cache_resource
def load_feature_model(model_type: str):
    paths = {
        "VGG19":       "feature_extractors/vgg19_feature_extractor.tflite",
        "EfficientNet":"feature_extractors/efficientnet_feature_extractor.tflite",
        "VGGFace":     "feature_extractors/vggface_feature_extractor_fc6.tflite"
    }
    p = paths.get(model_type)
    if not p or not os.path.exists(p):
        raise FileNotFoundError(f"Extractor for {model_type} not found: {p}")
    interp = tf.lite.Interpreter(model_path=p)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()

# 2) Load regressors
@st.cache_resource
def load_regressors():
    specs = {
        "VGG19-MLP":        "regressors/vgg19_ensemble_model.pkl",
        "EfficientNet-B3":  "regressors/Ridge_regressor.pkl",
        "VGGFace-Ensemble": "regressors/vggface_stacked_model.pkl"
    }
    models = {}
    for name, path in specs.items():
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        models[name] = joblib.load(path)
    return models

# 3) Preprocess
def preprocess_image(img: Image.Image, model_type: str):
    if model_type in ("VGG19","VGGFace"):
        size, fn = (224,224), vgg_preprocess
    else:
        size, fn = (300,300), eff_preprocess
    im = img.resize(size).convert("RGB")
    arr = img_to_array(im)[None,...]
    return fn(arr).astype(np.float32)

# 4) Predict BMI
def predict_bmi(img, interp, in_det, out_det, reg, gender_flag, model_type):
    # 1) extract features
    data = preprocess_image(img, model_type)
    interp.set_tensor(in_det[0]["index"], data)
    interp.invoke()
    feats = interp.get_tensor(out_det[0]["index"])  # shape (1, N)

    # 2) append gender features
    if model_type == "VGGFace":
        # stacking SVR expects 512 + 2 = 514
        gv = np.array([[1, 0]], dtype=np.float32) if gender_flag == 1 else np.array([[0, 1]], dtype=np.float32)
        feats = np.hstack([feats, gv])
    else:
        # dynamic logic for VGG19 or EfficientNet
        expected = getattr(reg, "n_features_in_", None)
        actual   = feats.shape[1]
        if expected and expected > actual:
            diff = expected - actual
            if diff == 2:
                gv = np.array([[1, 0]], dtype=np.float32) if gender_flag == 1 else np.array([[0, 1]], dtype=np.float32)
            elif diff == 1:
                gv = np.array([[gender_flag]], dtype=np.float32)
            else:
                raise ValueError(f"Regressor expects {expected} features but got {actual}")
            feats = np.hstack([feats, gv])

    # 3) predict
    return float(reg.predict(feats)[0])


def predict_bmi_from_frame(frame, *args):
    return predict_bmi(Image.fromarray(frame), *args)

class LiveBMI(VideoTransformerBase):
    def __init__(self, model_type, reg, gender_flag):
        self.interp, self.in_det, self.out_det = load_feature_model(model_type)
        self.reg           = reg
        self.gender_flag   = gender_flag
        self.model_type    = model_type

    def transform(self, frame):
        import cv2
        img = frame.to_ndarray(format="bgr24")
        try:
            bmi = predict_bmi_from_frame(
                img,
                self.interp, self.in_det, self.out_det,
                self.reg, self.gender_flag,
                self.model_type
            )
            cv2.putText(img, f"BMI: {bmi:.1f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        except:
            pass
        return img

def get_bmi_category(bmi: float) -> str:
    if bmi < 18.5: return "Underweight"
    if bmi < 25:   return "Normal weight"
    if bmi < 30:   return "Overweight"
    return "Obese"

# ── Streamlit UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="BMI Predictor", layout="centered")
st.title("🤳 Predict Your BMI!")

regressors = load_regressors()
workflow   = st.radio("Choose workflow:", ["By Gender", "By Model"])

if workflow == "By Gender":
    gender = st.selectbox("Select your gender:", ["", "Male", "Female"])
    if gender:
        gender_flag = 1 if gender == "Male" else 0
        model_type, model_key = (
            ("VGG19","VGG19-MLP") if gender=="Male"
            else ("EfficientNet","EfficientNet-B3")
        )
        st.markdown(f"**Pipeline:** `{model_type}` → `{model_key}`")

        reg_interp, in_det, out_det = load_feature_model(model_type)[0], *load_feature_model(model_type)[1:]
        reg = regressors[model_key]

        mode = st.radio("Input method:", ["Upload a photo","Take a photo","Live webcam"])
        if mode=="Live webcam":
            webrtc_streamer(
                key="live-bmi",
                video_transformer_factory=lambda: LiveBMI(model_type, reg, gender_flag),
                media_stream_constraints={"video": True, "audio": False}
            )
        else:
            img = None
            if mode=="Upload a photo":
                f = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp"])
                if f: img = Image.open(f)
            else:
                snap = st.camera_input("Take a photo")
                if snap: img = Image.open(snap)
            if img:
                st.image(img, caption="Your input", use_column_width=True)
                if st.button("🔍 Predict BMI"):
                    bmi = predict_bmi(img, *load_feature_model(model_type), regressors[model_key],
                                     gender_flag, model_type)
                    st.success(f"📏 Predicted BMI: {bmi:.1f}")
                    st.info(get_bmi_category(bmi))

else:
    choice = st.selectbox("Select model:", ["","VGG19-MLP","EfficientNet-B3","VGGFace-Ensemble"])
    if choice:
        # Determine flags
        if choice=="VGG19-MLP":
            model_type, gender_flag = "VGG19", None
        elif choice=="EfficientNet-B3":
            model_type, gender_flag = "EfficientNet", 0
        else:
            model_type, gender_flag = "VGGFace", 1  # VGGFace needs one-hot; flag=1 => male

        st.markdown(f"**Pipeline:** `{model_type}` → `{choice}`")

        reg = regressors[choice]
        interp, in_det, out_det = load_feature_model(model_type)

        mode = st.radio("Input method:", ["Upload a photo","Take a photo","Live webcam"])
        if mode=="Live webcam":
            webrtc_streamer(
                key="live-bmi",
                video_transformer_factory=lambda: LiveBMI(model_type, reg, gender_flag),
                media_stream_constraints={"video": True, "audio": False}
            )
        else:
            img = None
            if mode=="Upload a photo":
                f = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp"])
                if f: img = Image.open(f)
            else:
                snap = st.camera_input("Take a photo")
                if snap: img = Image.open(snap)
            if img:
                st.image(img, caption="Your input", use_column_width=True)
                if st.button("🔍 Predict BMI"):
                    bmi = predict_bmi(img, interp, in_det, out_det, reg, gender_flag, model_type)
                    st.success(f"📏 Predicted BMI: {bmi:.1f}")
                    st.info(get_bmi_category(bmi))
