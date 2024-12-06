import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import base64
import gdown
import os


# Function to download the model from Google Drive
def download_model():
    url = "https://drive.google.com/uc?export=download&id=1fftXlxFl-LPcA3SGENUwwVxyQu1uNWNK"  # Your model's direct download link
    output_path = 'plant_disease_vgg16_model.h5'
    gdown.download(url, output_path, quiet=False)

# Load model with caching to avoid re-downloading
def load_model_from_drive():
    if not os.path.exists('plant_disease_vgg16_model.h5'):
        download_model()  # Download model if not already present
    return load_model('plant_disease_vgg16_model.h5')

# Load the model

# Custom CSS for enhanced styling
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Custom styling
st.set_page_config(
    page_title="Plant Health Analyzer", 
    page_icon="🌱", 
    layout="wide"
)

# Optional: Set background image (comment out if no background image)
# set_background('path_to_background_image.png')

# Custom CSS
st.markdown("""
<style>
.main-title {
    font-size: 48px;
    color: #2c3e50;
    text-align: center;
    font-weight: bold;
    margin-bottom: 30px;
}
.subtitle {
    font-size: 20px;
    color: #7f8c8d;
    text-align: center;
    margin-bottom: 30px;
}
.upload-box {
    background-color: rgba(255, 255, 255, 0.8);
    border: 2px dashed #3498db;
    border-radius: 15px;
    padding: 30px;
    text-align: center;
    transition: all 0.3s ease;
}
.upload-box:hover {
    background-color: rgba(255, 255, 255, 0.9);
    border-color: #2980b9;
    box-shadow: 0 0 15px rgba(52, 152, 219, 0.3);
}
.prediction-card {
    background-color: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.stButton>button {
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 16px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #2980b9;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# Load the model and class indices
@st.cache_resource
def load_resources():
    model = load_model_from_drive()
    
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    class_labels = {v: k for k, v in class_indices.items()}  # Invert the class indices
    
    return model, class_labels

model, class_labels = load_resources()

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))                       # Resize image to match model input
    img_array = img_to_array(img) / 255.0                # Convert to array and normalize
    img_array = np.expand_dims(img_array, axis=0)        # Expand dimensions to match model input shape
    return img_array

# Function to make a prediction with confidence
def predict_image_class(img_array):
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    return predicted_class, confidence

# Streamlit interface
def main():
    # Title and Subtitle
    st.markdown('<h1 class="main-title">🌱 Plant Health Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Detect plant diseases with advanced AI technology</p>', unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # File Uploader
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a plant leaf image", 
            type=["jpg", "jpeg", "png"], 
            help="Upload a clear image of a plant leaf"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Display uploaded image and prediction
    if uploaded_file is not None:
        # Open and resize image
        image = Image.open(uploaded_file)
        max_size = (500, 500)
        image.thumbnail(max_size, Image.LANCZOS)

        with col2:
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Predict button with improved styling
            if st.button("Analyze Leaf", type="primary"):
                # Preprocess and predict
                img_array = preprocess_image(image)
                predicted_class, confidence = predict_image_class(img_array)

                # Prediction card with styling
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.markdown(f"### Prediction: {predicted_class}")
                st.markdown(f"### Confidence: {confidence:.2f}%")

                # Add context based on prediction
                if "Blight" in predicted_class:
                    st.warning("🚨 Potential disease detected. Recommend consulting an agricultural expert.")
                else:
                    st.success("✅ Leaf appears healthy. Continue good plant care practices!")
                st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
