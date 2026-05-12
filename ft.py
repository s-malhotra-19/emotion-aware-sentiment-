
import streamlit as st
from transformers import pipeline

# Page Configuration
st.set_page_config(page_title="Emotion-Aware Analysis", page_icon="🎭")

st.title("🎭 Emotion-Aware Sentiment Analysis")
st.markdown("Enter text below to detect the underlying emotion using a DistilBERT Transformer model.")

# Load the model using Pipeline (the easiest way for inference)
@st.cache_resource
def load_emotion_pipeline():
    # If you haven't trained locally yet, use a pre-trained emotion model from HF:
    # "bhadresh-savani/distilbert-base-uncased-emotion"
    try:
        pipe = pipeline("text-classification", model="./final_model", top_k=1)
    except:
        # Fallback to a high-quality public emotion model if local one isn't found
        pipe = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=1)
    return pipe

classifier = load_emotion_pipeline()

# User Input
user_input = st.text_area("What's on your mind?", placeholder="Type here...")

if st.button("Analyze Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            results = classifier(user_input)
            prediction = results[0][0]
            label = prediction['label'].upper()
            score = prediction['score'] * 100

            # UI Display
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Emotion", label)
            
            with col2:
                st.metric("Confidence Score", f"{score:.2f}%")

            # Progress bar for visual feedback
            st.progress(score / 100)
            
            if score > 70:
                st.success(f"The model is quite confident this text expresses {label}.")
            else:
                st.info("The model detected this emotion, but with lower confidence.")

st.sidebar.info("This project uses DistilBERT fine-tuned on the GoEmotions dataset.")