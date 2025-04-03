import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing import sequence

# Set page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# Load the saved model - using a try-except to handle import errors gracefully
@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('imdb_sentiment_model.keras')
        return model
    except ImportError as e:
        st.error(f"Error importing TensorFlow: {e}")
        st.info("""
        TensorFlow import error detected. This could be due to:
        1. Python 3.12 compatibility issues with TensorFlow
        2. Missing Visual C++ Redistributable on Windows
        3. TensorFlow installation issues
        
        Try these solutions:
        - Install Visual C++ Redistributable: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
        - Downgrade to Python 3.10 or 3.11 which have better TensorFlow support
        - Reinstall TensorFlow: `pip install tensorflow==2.15.0`
        """)
        return None

# Load word index mapping
@st.cache_resource
def load_word_index():
    try:
        with open('word_index.pkl', 'rb') as f:
            word_index_data = pickle.load(f)
        return word_index_data['word_index']
    except FileNotFoundError:
        st.error("Word index file not found. Please run the training script first.")
        return None

# Text preprocessing function using Keras sequence module if available
def preprocess_text(text, word_index, max_len=500, max_features=10000):
    # Tokenize the text
    words = text.lower().split()
    # Convert words to indices
    sequence_data = [word_index.get(word, 0) for word in words]
    # Filter indices larger than max_features
    sequence_data = [idx if idx < max_features else 0 for idx in sequence_data]
    
    try:
        # Try to use Keras sequence padding
        from tensorflow.keras.preprocessing import sequence
        padded_sequence = sequence.pad_sequences([sequence_data], maxlen=max_len)
    except ImportError:
        # Manual padding if Keras is not available
        if len(sequence_data) > max_len:
            padded_sequence = np.array([sequence_data[:max_len]])
        else:
            padded_sequence = np.array([[0] * (max_len - len(sequence_data)) + sequence_data])
    
    return padded_sequence

# Main function
def main():
    # App title and description
    st.title("🎬 IMDB Sentiment Analysis")
    st.markdown("""
    This app predicts whether a movie review is positive or negative.
    Enter your review in the text area below and click 'Analyze' to see the prediction.
    """)
    
    # Load the model and word index
    model = load_model()
    word_index = load_word_index()
    
    if model is None or word_index is None:
        st.stop()
    
    # Text input
    review = st.text_area("Enter your movie review here:", height=150)
    
    # Example reviews
    with st.expander("Example reviews"):
        st.markdown("""
        **Positive example:**
        I absolutely loved this movie! The acting was superb and the storyline kept me engaged throughout.
        
        **Negative example:**
        This was a complete waste of time. The plot was confusing and the characters were poorly developed.
        """)
    
    # Analyze button
    if st.button("Analyze Sentiment"):
        if review:
            # Display a spinner while processing
            with st.spinner("Analyzing..."):
                try:
                    # Preprocess the text
                    processed_review = preprocess_text(review, word_index)
                    
                    # Make prediction
                    prediction = model.predict(processed_review)[0][0]
                    
                    # Display result
                    st.subheader("Sentiment Analysis Result")
                    
                    # Create a progress bar for visualization
                    sentiment_score = float(prediction)
                    
                    # Display the sentiment score
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Sentiment Score", f"{sentiment_score:.2f}")
                    with col2:
                        if sentiment_score >= 0.5:
                            st.success("This review is POSITIVE 😊")
                        else:
                            st.error("This review is NEGATIVE 😞")
                    
                    # Visualization
                    st.progress(sentiment_score)
                    
                    # Confidence explanation
                    st.info(f"""
                    The model is {abs(sentiment_score - 0.5) * 2:.0%} confident in this prediction.
                    Scores close to 1.0 indicate strong positive sentiment, while scores close to 0.0 indicate strong negative sentiment.
                    """)
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.info("This could be due to TensorFlow compatibility issues. Please check the error message for details.")
        else:
            st.warning("Please enter a review to analyze.")
    
    # Add information about the model
    st.sidebar.header("About")
    st.sidebar.info("""
    This app uses a SimpleRNN model trained on the IMDB movie reviews dataset.
    
    The model was trained on 25,000 movie reviews and achieved an accuracy of approximately 85% on the test set.
    
    **Model Architecture:**
    - Embedding Layer (10,000 vocabulary size)
    - SimpleRNN Layer (64 units)
    - Dense Layer with Sigmoid Activation
    """)
    
    # Add troubleshooting section
    st.sidebar.header("Troubleshooting")
    st.sidebar.info("""
    **If you encounter TensorFlow errors:**
    
    1. TensorFlow has known issues with Python 3.12 on Windows
    2. Try using Python 3.10 or 3.11 instead
    3. Install the Microsoft Visual C++ Redistributable
    4. Make sure your TensorFlow version is compatible with your Python version
    """)

if __name__ == "__main__":
    main()