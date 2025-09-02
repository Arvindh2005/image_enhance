import streamlit as st
import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import Image
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim


PIXEL = 256

model_low = tf.keras.models.load_model("real_mass.h5")
model_mid = tf.keras.models.load_model("real_dream.h5")
model_bright = tf.keras.models.load_model("finalmass.h5")

def extract_test_input(image):
    image = np.array(image)
    #img = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    img = cv.resize(image, (PIXEL, PIXEL))
    return img.reshape(1, PIXEL, PIXEL, 3), img

def compute_brightness_uint8(image):
    brightness = np.mean(image, axis=-1, keepdims=True)
    brightness = brightness / 255.0
    return brightness

def fuse_outputs_uint8(input_img):
    img_float = input_img.astype(np.float32)

    out_low = model_low.predict(img_float)
    out_mid = model_mid.predict(img_float)
    out_bright = model_bright.predict(img_float)

    brightness = compute_brightness_uint8(input_img)

    mask_low = np.clip(1.0 - brightness * 3, 0.0, 1.0)
    mask_bright = np.clip((brightness - 0.5) * 2, 0.0, 1.0)
    mask_mid = 1.0 - np.clip(mask_low + mask_bright, 0.0, 1.0)

    total = mask_low + mask_mid + mask_bright + 1e-8
    mask_low /= total
    mask_mid /= total
    mask_bright /= total

    fused = (out_low * mask_low +
             out_mid * mask_mid +
             out_bright * mask_bright)

    return np.clip(fused.reshape(PIXEL, PIXEL, 3), 0, 255).astype(np.uint8)

def enhance_image(image):
    img_input, _ = extract_test_input(image)
    enhanced = fuse_outputs_uint8(img_input)
    return enhanced

st.markdown("""
    <style>
    .title {
        font-size: 72px;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 18px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">Low-Light Image Enhancer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enhance your Low Light Images!</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a low-light image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)

    if st.button("Enhance Image"):
        with st.spinner("Enhancing Image with model.."):
            time.sleep(1)
            enhanced_img = enhance_image(image)

        st.success("Image enhanced!")
        st.image(enhanced_img, caption="Enhanced Image", use_container_width=True)

        enhanced_pil = Image.fromarray(enhanced_img)
        st.download_button(
            label="📥 Download Enhanced Image",
            data=enhanced_pil.tobytes(),
            file_name="fused_enhanced.png",
            mime="image/png"
        )

    if st.checkbox("🔍 Compare Side-by-Side"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_container_width=True)
        with col2:
            enhanced_img = enhance_image(image)
            st.image(enhanced_img, caption="Enhanced Image", use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)


