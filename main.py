import pandas as pd
from datetime import datetime
import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from fpdf import FPDF
from io import BytesIO
import base64
import tempfile




# Function to set a background image
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:n7.png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call the function to set the background
set_background("n7.png")  # Replace with your background image file


# Set the working directory and paths to your model and class indices
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)


# Loading the class names (diseases)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))


# Precautions, Symptoms, and Treatment suggestions (Organic & Inorganic)
disease_info = {
    'Apple___Apple_scab': {
        'precautions': 'Prune infected branches, use fungicides like sulfur, and remove fallen leaves to prevent further spread.',
        'symptoms': 'Dark, sunken lesions on leaves, fruit, and stems. Yellowing of leaves.',
        'organic_treatment': 'Use neem oil or compost tea.',
        'inorganic_treatment': 'Use sulfur-based fungicides.',
    },
    'Apple___Black_rot': {
        'precautions': 'Remove infected fruit and leaves, prune infected branches, and apply copper-based fungicides.',
        'symptoms': 'Black lesions with concentric rings on leaves and fruit.',
        'organic_treatment': 'Use garlic oil spray.',
        'inorganic_treatment': 'Use copper hydroxide fungicides.',
    },
    'Apple___Cedar_apple_rust': {
        'precautions': 'Remove infected leaves and fruit, improve air circulation, and use resistant varieties.',
        'symptoms': 'Orange or red spots on leaves, and rust-colored lesions on fruit.',
        'organic_treatment': 'Use neem oil to treat fungal growth.',
        'inorganic_treatment': 'Use mancozeb or sulfur-based fungicides.',
    },
    'Apple___healthy': {
        'precautions': 'Ensure proper watering, fertilization, and soil care.',
        'symptoms': 'No symptoms present.',
        'organic_treatment': 'Use neem oil or compost tea.',
        'inorganic_treatment': 'N/A',
    },
    'Blueberry___healthy': {
        'precautions': 'Ensure proper care including pruning, watering, and good soil conditions.',
        'symptoms': 'No symptoms present.',
        'organic_treatment': 'N/A',
        'inorganic_treatment': 'Use azoxystrobin-based fungicides',
    },
    'Cherry_(including_sour)_Powdery_mildew': {
        'precautions': 'Apply fungicides, prune affected branches, and maintain good air circulation.',
        'symptoms': 'White powdery coating on leaves and fruit.',
        'organic_treatment': 'Use sulfur or baking soda solutions.',
        'inorganic_treatment': 'Use azoxystrobin-based fungicides.',
    },
    'Cherry_(including_sour)_healthy': {
        'precautions': 'Ensure good drainage, proper watering, and soil pH.',
        'symptoms': 'No symptoms present.',
        'organic_treatment': 'N/A',
        'inorganic_treatment': 'N/A',
    },
    'Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot': {
        'precautions': 'Use fungicides, avoid overhead irrigation, and practice crop rotation.',
        'symptoms': 'Grayish, rectangular lesions with yellow borders on leaves.',
        'organic_treatment': 'Use neem oil or compost tea.',
        'inorganic_treatment': 'Use chlorothalonil or copper-based fungicides.',
    },
    'Corn_(maize)Common_rust': {
        'precautions': 'Use resistant corn varieties, apply fungicides, and remove affected plants.',
        'symptoms': 'Rust-colored pustules on the upper side of leaves.',
        'organic_treatment': 'Use neem oil or a garlic spray.',
        'inorganic_treatment': 'Use propiconazole fungicides.',
    },
    'Corn_(maize)_Northern_Leaf_Blight': {
        'precautions': 'Use resistant varieties, remove infected leaves, and apply fungicides.',
        'symptoms': 'Long, rectangular lesions with gray centers and dark borders on leaves.',
        'organic_treatment': 'Use compost tea or neem oil.',
        'inorganic_treatment': 'Use chlorothalonil or mancozeb fungicides.',
    },
    'Corn_(maize)_healthy': {
        'precautions': 'Ensure proper irrigation, soil fertility, and pest management.',
        'symptoms': 'No symptoms present.',
        'organic_treatment': 'Use neem oil or a garlic spray.',
        'inorganic_treatment': 'Use azoxystrobin-based fungicides',
    },
    'Grape___Black_rot': {
        'precautions': 'Remove infected leaves and fruit, apply fungicides, and prune infected vines.',
        'symptoms': 'Dark, sunken lesions on leaves, fruit, and stems.',
        'organic_treatment': 'Use sulfur or neem oil.',
        'inorganic_treatment': 'Use copper-based fungicides.',
    },
    'Grape__Esca(Black_Measles)': {
        'precautions': 'Prune infected vines, remove affected leaves, and apply fungicides.',
        'symptoms': 'Black lesions and discoloration on leaves and vines.',
        'organic_treatment': 'Use garlic oil or compost tea.',
        'inorganic_treatment': 'Use Bordeaux mixture or sulfur fungicides.',
    },
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': {
        'precautions': 'Remove infected leaves, apply fungicides, and improve air circulation.',
        'symptoms': 'Small, dark lesions with yellow halos on leaves.',
        'organic_treatment': 'Use neem oil or organic fungicides.',
        'inorganic_treatment': 'Use chlorothalonil or copper-based fungicides.',
    },
    'Grape___healthy': {
        'precautions': 'Ensure proper soil, irrigation, and sunlight.',
        'symptoms': 'No symptoms present.',
        'organic_treatment': 'Use garlic oil or compost tea.',
        'inorganic_treatment': 'Use azoxystrobin-based fungicides',
    },
    'Orange__Haunglongbing(Citrus_greening)': {
        'precautions': 'Remove infected plants, control insect vectors, and apply citrus greening management strategies.',
        'symptoms': 'Yellowing leaves, misshapen fruit, and stunted growth.',
        'organic_treatment': 'Use organic insecticides and citrus grease.',
        'inorganic_treatment': 'Use systemic insecticides like imidacloprid.',

    },
    'Peach___Bacterial_spot': {
        'precautions': 'Prune infected branches, apply copper-based fungicides, and avoid overhead irrigation.',
        'symptoms': 'Water-soaked lesions, yellow halos, and bacterial ooze on fruit.',
        'organic_treatment': 'Use copper sulfate spray.',
        'inorganic_treatment': 'Use streptomycin or copper-based fungicides.',
    },
    'Peach___healthy': {
        'precautions': 'Ensure well-draining soil and regular pruning.',
        'symptoms': 'No symptoms present.',
        'organic_treatment': 'Use neem oil or copper sulfate spray.',
        'inorganic_treatment': 'Use copper hydroxide or streptomycin fungicides.',
    },
    'Pepper,bell__Bacterial_spot': {
        'precautions': 'Remove infected plants, avoid overhead irrigation, and use copper-based bactericides.',
        'symptoms': 'Water-soaked spots, sunken lesions on leaves, and fruit.',
        'organic_treatment': 'Use neem oil or copper sulfate spray.',
        'inorganic_treatment': 'Use copper hydroxide or streptomycin fungicides.',
    },
    'Pepper,bell__healthy': {
        'precautions': 'Ensure proper spacing, regular watering, and good soil conditions.',
        'symptoms': 'No symptoms present.',
        'organic_treatment': 'Use organic insecticides and citrus grease.',
        'inorganic_treatment': 'Use systemic insecticides like imidacloprid',
    },
}

# File to store history
HISTORY_FILE = "classification_history.csv"

# Function to save classification results to the history file
def save_to_history(prediction, image_name, timestamp):
    record = {
        "Timestamp": [timestamp],
        "Image Name": [image_name],
        "Prediction": [prediction]
    }
    record_df = pd.DataFrame(record)

    # Check if the history file exists
    if os.path.exists(HISTORY_FILE):
        history_df = pd.read_csv(HISTORY_FILE)
        updated_df = pd.concat([history_df, record_df], ignore_index=True)
    else:
        updated_df = record_df

    updated_df.to_csv(HISTORY_FILE, index=False)


# Function to retrieve history
def get_history():
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE)
        except Exception as e:
            st.warning(f"Error loading history file: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

    
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
    
    # Get information based on the predicted class
    info = disease_info.get(predicted_class_name, {
        'precautions': 'No precautions available.',
        'symptoms': 'No symptoms available.',
        'organic_treatment': 'No organic treatment available.',
        'inorganic_treatment': 'No inorganic treatment available.',
    })
    
    return predicted_class_name, info

def generate_pdf_report(prediction, info, image):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt="Plant Disease Classification Report", ln=True, align='C')
    pdf.ln(10)
    
    # Add the uploaded image
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
        temp_image_path = temp_image.name
        image.save(temp_image_path)  # Save the image temporarily
        pdf.image(temp_image_path, x=10, y=40, w=80)  # Place the image on the left
    
    # Add the text beside the image
    pdf.set_xy(100, 40)  # Start position for the text (to the right of the image)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(100, 10, txt=f"Disease Detected: {prediction}", border=0)
    
    # Move cursor for detailed information below
    pdf.set_xy(10, 130)  # Adjust y-position to below the image and text
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(200, 10, txt="Details:", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=f"Precautions: {info['precautions']}\n")
    pdf.multi_cell(0, 10, txt=f"Symptoms: {info['symptoms']}\n")
    pdf.multi_cell(0, 10, txt=f"Organic Treatment: {info['organic_treatment']}\n")
    pdf.multi_cell(0, 10, txt=f"Inorganic Treatment: {info['inorganic_treatment']}\n")
    
    # Save PDF to BytesIO
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    
    # Clean up temporary file
    os.unlink(temp_image_path)
    
    return pdf_output


# Adding CSS for the simple title transition
# Adding CSS for the title styling and centering
st.markdown("""
    <style>
    html {
      height: 100%;
    }
    body {
      display: flex;
      height: 100%;
      background-color: #333;
    }
    
    /* Simplified transition and centering for the title */
    .title {
      margin: auto;
      color: white;
      font: 700 normal 2.5em 'Tahoma';
      text-shadow: 5px 2px #222324, 2px 4px #222324, 3px 5px #222324;
      opacity: 0;
      animation: fadeIn 2s forwards;
      text-align: center; /* Center text horizontally */
    }

    /* Fade-in effect */
    @keyframes fadeIn {
      from {
        opacity: 0;
      }
      to {
        opacity: 1;
      }
    }

    /* Style for expanders (Precautions, Symptoms, etc.) */
    .expander-header {
      background-color: #0044cc;  /* Blue background */
      color: white;
      font-weight: bold;
      padding: 10px;
      border-radius: 5px;
    }

    .expander-content {
      background-color: #f0f8ff;  /* Light blue background */
      padding: 10px;
      border-radius: 5px;
      color: #333;
    }

    .expander-content div {
      margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Adding the title HTML wrapped in the container
st.markdown('<div class="title-container"><div class="title">PHYTOSCAN</div></div>', unsafe_allow_html=True)


# Adding JavaScript for the typing animation and flicker effect

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)


    # Display the uploaded image in one column
    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction, info = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_to_history(prediction, uploaded_image.name, timestamp)
            
            # Apply animation to sections
            with st.expander("Precautions:", expanded=True):
                st.markdown(f'<div class="expander-header">Precautions:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="expander-content">{info["precautions"]}</div>', unsafe_allow_html=True)
            
            with st.expander("Symptoms:", expanded=True):
                st.markdown(f'<div class="expander-header">Symptoms:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="expander-content">{info["symptoms"]}</div>', unsafe_allow_html=True)
            
            with st.expander("Organic Treatment:", expanded=True):
                st.markdown(f'<div class="expander-header">Organic Treatment:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="expander-content">{info["organic_treatment"]}</div>', unsafe_allow_html=True)
            
            with st.expander("Inorganic Treatment:", expanded=True):
                st.markdown(f'<div class="expander-header">Inorganic Treatment:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="expander-content">{info["inorganic_treatment"]}</div>', unsafe_allow_html=True)
            
            # Generate download button for the report
            pdf_data = generate_pdf_report(prediction, info,image)
            st.download_button(
                label="Download Report as PDF",
                data=pdf_data,
                file_name="disease_report.pdf",
                mime="application/pdf"
            ) 
if st.button("View History"):
    history_df = get_history()
    if not history_df.empty:
        st.write("### Classification History")
        st.dataframe(history_df)  # Display the history in a table
    else:
        st.info("No history available yet.")
if st.button("Clear History"):
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        st.success("History has been cleared.")
    else:
        st.info("No history found.")