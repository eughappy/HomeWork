import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Load models
simple_cnn_model = load_model("D:\\Repos\\StreamlitHW16\\simple_cnn_model.h5")
vgg16_model = load_model("D:\\Repos\\StreamlitHW16\\vgg16_model.h5")
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# Define function to preprocess images
def preprocess_image(image, target_size, model_type):
    if model_type == 'Simple CNN':
        # Перетворення на чорно-біле зображення
        image = image.convert('L')
        image = ImageOps.invert(image)
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Define function to plot loss and accuracy
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history['accuracy'], label='Training Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    st.pyplot(fig)

# Streamlit UI
st.title('Image Classification with Neural Networks')
st.write("Upload an image to classify it using the trained models.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

model_choice = st.selectbox('Select a model', ('Simple CNN', 'VGG16'))

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    if model_choice == 'Simple CNN':
        model = simple_cnn_model
        target_size = (28, 28)
    else:
        model = vgg16_model
        target_size = (32, 32)
    
    processed_image = preprocess_image(image, target_size, model_choice)
    
    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Display prediction
    st.write(f"Predicted class: {class_names[predicted_class[0]]}")
    st.write("Class probabilities:")
    st.write(prediction)
    
    # Plot loss and accuracy
    if model_choice == 'Simple CNN':
        history = np.load("D:\\Repos\\StreamlitHW16\\simple_cnn_history.npy", allow_pickle=True).item()
    else:
        history = np.load("D:\\Repos\\StreamlitHW16\\vgg16_history.npy", allow_pickle=True).item()
    
    plot_history(history)
