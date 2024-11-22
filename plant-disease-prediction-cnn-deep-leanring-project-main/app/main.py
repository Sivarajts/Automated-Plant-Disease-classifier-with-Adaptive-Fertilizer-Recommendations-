import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Load disease descriptions and fertilizer recommendations
disease_info = {

    'Apple___Apple_scab': {
        'description': 'Fungal disease causing dark, scabby lesions on leaves and fruit.',
        'fertilizer': 'Use a balanced N-P-K fertilizer (10-10-10) and sulfur-based fungicides like Bonide Sulfur Plant Fungicide.'
    },
    'Apple___Black_rot': {
        'description': 'Causes dark, rotting lesions on fruit, leaves, and bark.',
        'fertilizer': 'Apply nitrogen-rich fertilizer like Urea (46-0-0) to promote healthy growth.'
    },
    'Apple___Cedar_apple_rust': {
        'description': 'Causes yellow-orange spots on apple leaves, affecting fruit quality.',
        'fertilizer': 'Use a balanced fertilizer like 10-10-10, along with a fungicide such as Immunox Multi-Purpose Fungicide.'
    },
    'Apple___healthy': {
        'description': 'The plant is in good health, free from disease.',
        'fertilizer': 'Apply a balanced fertilizer like 12-12-12, focusing on potassium-rich blends such as Potassium Sulfate (0-0-50).'
    },
    'Blueberry___healthy': {
        'description': 'Healthy plant showing no signs of disease.',
        'fertilizer': 'Apply acidic fertilizers such as Ammonium Sulfate (21-0-0) or Holly-tone (4-3-4).'
    },
    'Cherry (including sour)___healthy': {
        'description': 'Healthy cherry tree, no signs of disease.',
        'fertilizer': 'Use a nitrogen-based fertilizer like Blood Meal (12-0-0) or a balanced 5-10-10 blend.'
    },
    'Cherry (including sour)___Powdery_mildew': {
        'description': 'Fungal disease causing white powdery coating on leaves and fruit.',
        'fertilizer': 'Apply a balanced fertilizer like 10-10-10, and use Bonide Sulfur Fungicide.'
    },
    'Corn (maize)___Cercospora_leaf_spot_Gray_leaf_spot': {
        'description': 'Fungal spots reduce photosynthesis and yield.',
        'fertilizer': 'Use potassium-rich fertilizers like Potassium Sulfate (0-0-50) to strengthen plant resistance.'
    },
    'Corn (maize)___Common_rust': {
        'description': 'Rust-colored pustules appear on leaves, reducing yield.',
        'fertilizer': 'Apply nitrogen-rich fertilizers like Urea (46-0-0) or Ammonium Nitrate (34-0-0).'
    },
    'Corn (maize)___healthy': {
        'description': 'No visible signs of disease.',
        'fertilizer': 'Use nitrogen-rich fertilizers like Ammonium Sulfate (21-0-0).'
    },
    'Corn (maize)___Northern_Leaf_Blight': {
        'description': 'Large, grayish lesions form on leaves, reducing yield.',
        'fertilizer': 'Apply balanced fertilizers like 10-10-10, and use fungicides like Daconil.'
    },
    'Grape___Black_rot': {
        'description': 'Causes dark, shriveled fruit and leaf spots.',
        'fertilizer': 'Use a balanced N-P-K fertilizer like 10-10-10 and fungicides such as Mancozeb.'
    },
    'Grape___Esca (Black Measles)': {
        'description': 'Streaks and spots on grapes, reducing fruit quality.',
        'fertilizer': 'Apply organic fertilizers like compost or fish emulsion.'
    },
    'Grape___healthy': {
        'description': 'No disease present; the plant is healthy.',
        'fertilizer': 'Use a balanced fertilizer like 10-10-10 in spring.'
    },
    'Grape___Leaf_blight (Isariopsis_Leaf_Spot)': {
        'description': 'Causes brown lesions on leaves, affecting grape production.',
        'fertilizer': 'Apply potassium-rich fertilizer like Potassium Sulfate (0-0-50) and fungicide like Captan.'
    },
    'Orange___Haunglongbing (Citrus Greening)': {
        'description': 'Bacterial infection causing yellowing leaves and poor-quality fruit.',
        'fertilizer': 'Apply micronutrient-rich fertilizers like Citrus-tone (5-2-6) with added zinc, manganese, and iron.'
    },
    'Peach___Bacterial_spot': {
        'description': 'Bacterial infections leading to spots on leaves and fruit.',
        'fertilizer': 'Use a nitrogen-rich fertilizer like Ammonium Nitrate (34-0-0) and apply Copper Fungicide.'
    },
    'Peach___healthy': {
        'description': 'The plant is free from disease.',
        'fertilizer': 'Apply a balanced fertilizer like 8-8-8 in the spring to support fruit production.'
    },
    'Pepper, bell___Bacterial_spot': {
        'description': 'Bacterial spots on leaves and fruit, reducing yield.',
        'fertilizer': 'Apply phosphorus and potassium-rich fertilizers like 0-10-10 and Copper Fungicide.'
    },
    'Pepper, bell___healthy': {
        'description': 'The plant is healthy with no visible disease symptoms.',
        'fertilizer': 'Use a balanced fertilizer like 10-10-10.'
    },
    'Potato___Early_blight': {
        'description': 'Dark, concentric spots on leaves, reducing yield.',
        'fertilizer': 'Apply potassium-rich fertilizer like 0-0-50 and fungicides like Daconil.'
    },
    'Potato___healthy': {
        'description': 'Healthy potato plant.',
        'fertilizer': 'Use a balanced fertilizer like 15-15-15, with extra nitrogen for tuber growth.'
    },
    'Potato___Late_blight': {
        'description': 'Water-soaked lesions on leaves, leading to tuber rot.',
        'fertilizer': 'Apply balanced fertilizers like 10-10-10, and use a fungicide like Chlorothalonil.'
    },
    'Strawberry___healthy': {
        'description': 'Healthy strawberry plant.',
        'fertilizer': 'Apply balanced fertilizers such as 5-10-10 with organic compost.'
    },
    'Strawberry___Leaf_scorch': {
        'description': 'Purple spots on leaves, leading to drying and dieback.',
        'fertilizer': 'Use potassium-rich fertilizers like 0-0-50 and ensure proper watering.'
    },
    'Tomato___Bacterial_spot': {
        'description': 'Bacterial infections causing dark spots on leaves and fruit.',
        'fertilizer': 'Apply nitrogen-based fertilizers like Ammonium Nitrate (34-0-0) and Copper Fungicide.'
    },
    'Tomato___Early_blight': {
        'description': 'Circular lesions on leaves, leading to defoliation.',
        'fertilizer': 'Use balanced fertilizers like 10-10-10, and control with Daconil.'
    },
    'Tomato___healthy': {
        'description': 'Healthy tomato plant.',
        'fertilizer': 'Apply nitrogen-rich fertilizer like Ammonium Sulfate (21-0-0).'
    },
    'Tomato___Late_blight': {
        'description': 'Water-soaked spots on leaves and fruit, leading to rot.',
        'fertilizer': 'Use balanced fertilizers like 10-10-10 and fungicides like Chlorothalonil.'
    }
}





working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.title('Leaf Disease Detection')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')

            # Fetch and display disease description and fertilizer recommendation
            disease_details = disease_info.get(str(prediction), {})
            description = disease_details.get('description', 'No description available.')
            fertilizer = disease_details.get('fertilizer', 'No fertilizer recommendation available.')

            st.write(f"**Disease Description:** {description}")
            st.write(f"**Recommended Fertilizer:** {fertilizer}")
