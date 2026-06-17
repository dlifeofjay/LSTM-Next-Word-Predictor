import streamlit as st
import json
from src.architecture import WordPred
import torch
import torch.nn as nn
import string
from PIL import Image

st.set_page_config(page_title="LSTM Next Word Predictor", layout="centered")

# Load artifacts

model = WordPred(2181)
loaded_weights = torch.load(r"Artifacts/mini_lm.pt", weights_only=True)
model.load_state_dict(loaded_weights)
model.eval()

with open(r"Artifacts/tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer = json.load(f)

# Streamlit UI

st.title("LSTM - Next Word Predictor (NLP)")
st.write("This project simulates early language model using LSTM sequential modelling. "
         "Tokenizer and main model architecture was built from scratch.")

st.divider()

st.subheader("Generate Text")
st.write("Enter words relating to machine learning, LSTM and the "
         "*Attention Is All You Need* paper — the model tries to complete the sentence.")

# Clean text

def clean_text(text):
    """
    This function aims to clean the text extracted
    """
    clean = str.maketrans('', '', string.punctuation)
    text = text.translate(clean)
    text = text.replace("\n", " ").lower().split()

    # Expecting texts splitted in list
    return text

# Convert them to tokens

def prediction(text, word_to_id, sequence):
    """
    Take cleaned text, convert to tokens,
    predict and add the predicted word to
    sentence in a loop
    """

    # convert texts to token
    tok = [word_to_id.get(tk, 0) for tk in text]

    token_to_word = {idx: word for word, idx in word_to_id.items()}

    # model prediction
    with torch.no_grad():
        for _ in range(sequence):
            # Take last 30 tokens (or all if less than 30)
            input_tokens = tok[-30:]
            
            # Convert to tensor with batch dimension: [1, seq_len]
            input_tensor = torch.tensor([input_tokens], dtype=torch.long)
            
            # Get prediction
            output = model(input_tensor)
            next_token_id = output.argmax(dim=1).item()
            
            if next_token_id == 0:        # UKN token
                break
                
            # Get the predicted word
            pred_word = token_to_word.get(next_token_id, "UKN")
            
            # Append to both lists
            text.append(pred_word)
            tok.append(next_token_id)
    
    return " ".join(text)

# Input
text = st.text_input("Enter your prompt", placeholder="e.g. Disadvantage of LSTM")

col1, col2 = st.columns([3, 1])
with col1:
    sequence = st.slider("Words to generate", min_value=5, max_value=50, value=30, step=5)
with col2:
    st.write("")  # spacing
    generate = st.button("Generate", use_container_width=True)

# When user clicks button
if generate:
    if text.strip():
        text_list = clean_text(text)
        result = prediction(text_list, tokenizer, sequence)
        st.success(result)
    else:
        st.error("Please write something first.")

st.divider()

# Learning curve
st.subheader("Training Loss Curve")

image = Image.open("Loss plot.png")
st.image(image, caption="Training Loss over Epochs")

# Analysis
with st.expander("Analysis and Next Steps"):
    st.write("**Key Observation:**")
    st.write("This model suffered from **severe overfitting** even after continuous hyperparameter tuning.")

    st.write("**Next Steps:**")
    st.write("- Gather much more text data for large scale training to improve accuracy")
    st.write("- More sophisticated text cleaning and preprocessing")
    st.write("- Use a more efficient open-source tokenizer")
