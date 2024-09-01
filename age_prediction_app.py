import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from PIL import Image

# Load the pre-trained model
@st.cache_resource
def load_trained_model(model_path):
    return load_model(model_path)

# Function to predict age from an image
def predict_age_from_image(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        st.error(f"Image not found or cannot be loaded: {image_path}")
        return None

    # Resize and display the image in a smaller size
    img = cv2.resize(img, (224, 224))
    img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    predicted_age = model.predict(img)
    return img_display, max(0, predicted_age[0][0])

# Streamlit UI setup
st.set_page_config(layout="wide")  # Use the entire width of the page

# Centered and catchy title with hero tag
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Age-Aura: discover your age🕒</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #555;'>Upload an Image and Let our model Guess Your Age!</h3>", unsafe_allow_html=True)

model_path = "trained_model.h5"  # Path to your trained model
model = load_trained_model(model_path)

# File uploader for image upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to a directory
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict age using the uploaded file
    img_display, predicted_age = predict_age_from_image(file_path, model)
    if predicted_age is not None:
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        # Display the image in the first column
        with col1:
            st.image(img_display, caption='Input Image', use_column_width=False, width=200)  # Reduced width
        
        # Display the predicted age in the second column
        with col2:
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.success(f"Predicted Age: {predicted_age:.2f}")

# Required functions
def load_images_and_labels(dataset_path, image_size=(224, 224), num_images=10):
    images = []
    labels = []
    for i, filename in enumerate(os.listdir(dataset_path)):
        if i >= num_images:
            break
        if filename.endswith('.jpg'):
            age = int(filename.split('_')[0])
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(age)
    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    images = images / 255.0
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
