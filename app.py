import subprocess
subprocess.run(["pip", "install", "tensorflow"])

import tensorflow as tf
import streamlit as st
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('cifar10_model.keras')

# Streamlit app
st.title("Image Classification App")

# Upload an image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(32, 32))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)  # Create a batch

    # Make predictions
    predictions = model.predict(image_array)
    score = tf.nn.softmax(predictions[0])

    # Display the results
    st.image(image, caption=f"Predicted class: {np.argmax(score)}, Confidence: {100 * np.max(score):.2f}%")
