import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os


# Load trained model & label encoder

@st.cache_resource
def load_models():
    rnn_model = load_model("rnn_emotion_model.h5")  # your saved RNN
    label_encoder = joblib.load("label_encoder.pkl")
    return rnn_model, label_encoder

rnn_model, label_encoder = load_models()


# Feature Extraction for RNN

def get_mfcc(file_path, sr=16000, n_mfcc=40):
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    audio, _ = librosa.effects.trim(audio)
    audio = audio / np.max(np.abs(audio))
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # shape -> (timesteps, n_mfcc)


# Streamlit UI

st.set_page_config(page_title="Emotion Recognition App", page_icon="🎤", layout="centered")

st.title("Speech Emotion Recognition (RNN)")
st.write("Upload a voice clip to predict the emotion using an LSTM-based RNN model.")

uploaded_file = st.file_uploader("Upload an audio file (wav/mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Extract features
    features = get_mfcc(temp_path)  # shape -> (timesteps, n_mfcc)

    # Pad or truncate to 200 time steps
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    features_padded = pad_sequences(
        [features],           # wrap in list to make it 3D: (samples, timesteps, n_mfcc)
        maxlen=200,           # must match your RNN input
        dtype='float32',
        padding='post',       # add zeros at the end if shorter
        truncating='post'     # cut extra frames if longer
    )

    # Predict
    preds = rnn_model.predict(features_padded)
    pred_class = np.argmax(preds, axis=1)[0]
    pred_label = label_encoder.inverse_transform([pred_class])[0]

    # Display results
    st.audio(temp_path, format="audio/wav")
    st.success(f"Predicted Emotion: **{pred_label}**")

    # Probability distribution
    st.write("### Prediction Probabilities")
    probs = dict(zip(label_encoder.classes_, preds[0]))
    st.bar_chart(probs)

    # Cleanup temp file
    os.remove(temp_path)
