import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import matplotlib.pyplot as plt

# Set the title and a brief description
st.set_page_config(page_title="Image Forgery Detection", page_icon="🔍")
st.title("🔍 Image Forgery Detection")
st.write(
    """
Welcome to the Image Forgery Detection app. 
Upload an image, and we will analyze it to detect any possible forgery using a deep learning model.
"""
)

# Load the pre-trained model
model = tf.keras.models.load_model("BEProjectMantraNetModel.h5")

# Define class names
class_names = ["Forged", "Not Forged"]

# Function to convert input image to ELA applied image
def convert_to_ela_image(path, quality):
    original_image = Image.open(path).convert("RGB")
    resaved_file_name = "resaved_image.jpg"
    original_image.save(resaved_file_name, "JPEG", quality=quality)
    resaved_image = Image.open(resaved_file_name)
    ela_image = ImageChops.difference(original_image, resaved_image)
    extrema = ela_image.getextrema()
    max_difference = max([pix[1] for pix in extrema])
    if max_difference == 0:
        max_difference = 1
    scale = 350.0 / max_difference
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

# Function to prepare image
def prepare_image(image_path):
    image_size = (224, 224)
    return (
        np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten()
        / 255.0
    )

# Display prediction result
def display_prediction(image_path, y_pred_class):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    original_image = plt.imread(image_path)
    ax[0].axis("off")
    ax[0].imshow(original_image)
    ax[0].set_title("Original Image")
    ax[1].axis("off")
    ax[1].imshow(convert_to_ela_image(image_path, 90))
    ax[1].set_title("ELA Image")
    st.pyplot(fig)
    st.markdown(f"### Detection: **{class_names[y_pred_class]}**")

# File uploader and detect button
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])
if uploaded_file is not None:
    st.image(
        uploaded_file,
        use_column_width=None,
        clamp=False,
        channels="RGB",
        output_format="auto",
    )
    st.write("")
    #st.write("Classifying...")

    # Prepare the image
    test_image = prepare_image(uploaded_file)

    # Display status
    status_text = st.empty()

    # Detect button
    if st.button("Detect Forgery"):
        with st.spinner("Analyzing the image..."):
            status_text.text("🔄 Predicting...")
            test_image = test_image.reshape(-1, 224, 224, 3)
            y_pred = model.predict(test_image)
            y_pred_class = round(y_pred[0][0])
        display_prediction(uploaded_file, y_pred_class)
        status_text.text("✅ Predicted")
else:
    st.info("Please upload an image file to get started.")
