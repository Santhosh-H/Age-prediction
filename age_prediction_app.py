import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the pre-trained model
@st.cache_resource
def load_trained_model(model_path):
    return load_model(model_path)

# Function to predict age from the dataset
def predict_age_from_dataset(model):
    dataset_path = './utkface-new/UTKFace'  # Update with your dataset path
    images, labels = load_images_and_labels(dataset_path)
    X_train, X_test, y_train, y_test = preprocess_data(images, labels)
    predictions = model.predict(X_test)
    return predictions, y_test

# Function to predict age from an image
def predict_age_from_image(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        st.error(f"Image not found or cannot be loaded: {image_path}")
        return None

    st.image(img, caption='Input Image', use_column_width=True)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    predicted_age = model.predict(img)
    return max(0, predicted_age[0][0])

# Streamlit UI setup
st.title("Age Prediction")
model_path = "trained_model.h5"  # Path to your trained model
model = load_trained_model(model_path)

# Button for predicting age from the dataset
if st.button("Predict Age from Dataset"):
    st.write("Running prediction on dataset...")
    predictions, actual_ages = predict_age_from_dataset(model)
    st.write(f"Predicted Ages: {predictions[:10]}")
    st.write(f"Actual Ages: {actual_ages[:10]}")

# Text input and button for predicting age from an image path
image_path = st.text_input("Enter image path")
if st.button("Predict Age from Image"):
    if image_path:
        predicted_age = predict_age_from_image(image_path, model)
        if predicted_age is not None:
            st.write(f"Predicted Age: {predicted_age}")
    else:
        st.error("Please enter a valid image path.")

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
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

