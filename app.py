import streamlit as st
import numpy as np
import tensorflow as tf
import os
from src.features import extract_mfcc

# Load the trained model
@st.cache_resource
def load_model():
    # If you changed your save format to .keras, update the filename here!
    return tf.keras.models.load_model('models/emotion_model.h5')

model = load_model()

# RAVDESS Emotion Labels
emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

# UI Layout
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a `.wav` audio file from the RAVDESS dataset to detect the emotion.")

# File uploader
uploaded_file = st.file_uploader("Choose a .wav file...", type="wav")

if uploaded_file is not None:
    # Let the user play the audio file in the browser
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Detect Emotion", type="primary"):
        # 1. Save the uploaded file temporarily so librosa can process it
        temp_path = os.path.join("data", "temp_audio.wav")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Analyzing audio frequencies..."):
            # 2. Extract features using the function we already built
            mfcc = extract_mfcc(temp_path)
            
            # 3. Reshape for the model: (time_steps, features) -> (1, time_steps, features)
            input_data = np.expand_dims(mfcc, axis=0)
            
            # 4. Predict
            predictions = model.predict(input_data)
            predicted_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_index]
            
            # 5. Display Result
            st.divider()
            st.success(f"### Detected Emotion: **{emotions[predicted_index]}**")
            st.info(f"**Confidence Score:** {confidence:.1%}")
            
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)