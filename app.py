import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Configuration
st.set_page_config(page_title="Next Word Prediction - LSTM", page_icon="📝", layout="centered")

# Load Model and Tokenizer
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("next_word_lstm.h5")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_assets()

# Predict Function
def predict_next_word(model, tokenizer, text, max_sequence_len=14):
    token_list = tokenizer.texts_to_sequences([text.lower()])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# User Interface
st.title("📝 Next Word Prediction App")
st.write("Shakespeare ke **Hamlet** text dataset par trained LSTM Model ka istemaal karke agla lafz predict karein.")

st.markdown("---")

user_input = st.text_input("Koi bhi sentence/text enter karein:", "You come most carefully")

if st.button("Predict Next Word", type="primary"):
    if user_input.strip() == "":
        st.warning("Kripya text enter karein!")
    else:
        with st.spinner("Predicting..."):
            next_word = predict_next_word(model, tokenizer, user_input)
            if next_word:
                st.success(f"**Predicted Next Word:** `{next_word}`")
                st.info(f"**Complete Sentence:** {user_input} **{next_word}**")
            else:
                st.error("Word predict nahi ho saka. Text badal kar try karein.")

st.markdown("---")
st.caption("Developed for Next Word Prediction Project using Deep Learning (LSTM) & Streamlit.")
