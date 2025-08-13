import streamlit as st
from ocr_model import smart_ocr_with_fallback
import tempfile
import os

st.set_page_config(page_title="OCR App", layout="centered")

st.title(" OCR App")
st.write("Upload an image and extract the text")

# Upload image
uploaded_file = st.file_uploader("📷 Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_image_path = tmp.name

    st.write("Extracting")
    result = smart_ocr_with_fallback(temp_image_path)
    st.success(f"✅ Final Extracted Result: {result}")

    # Optionally delete temp file after processing
    os.remove(temp_image_path)
