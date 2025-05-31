import streamlit as st
import joblib
from preprocess import clean_text
from visualizations import sentiment_scoring_lollipop_chart, sunburst_visualization
from collections import Counter

# Load Logistic Regression model and vectorizer
model = joblib.load('models/logistic_regression.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# App configuration
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="wide")

st.markdown("""
    <style>
    /* Global App Styles */
    .stApp {
        background: linear-gradient(135deg, #1C1F24, #23272B);  /* Darker theme with gradient */
        color: #F0F0F0;  /* Light color for better readability */
        font-family: 'Poppins', sans-serif;
        padding: 20px;
        line-height: 1.5;  /* Improve text readability */
        transition: background 0.5s ease;
    }

    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #252B30;  /* Darker sidebar */
        color: #F0F0F0;
        padding: 30px;
        border-right: 2px solid #4D5159;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.5);
        transition: background 0.5s ease;
    }

    .sidebar .sidebar-content h1 {
        font-size: 28px;
        font-weight: 700;  /* Bolder font for headers */
        color: #00ADB5;
        text-align: center;
        margin-bottom: 20px;  /* Space below the header */
    }

    .sidebar .sidebar-content a {
        color: #F0F0F0;
        text-decoration: none;
        transition: color 0.3s ease, transform 0.3s ease;
    }

    .sidebar .sidebar-content a:hover {
        color: #00ADB5;
        transform: scale(1.1);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00ADB5, #008F96);
        color: white;
        border-radius: 12px;
        font-size: 18px;
        padding: 16px 24px;
        border: none;
        transition: all 0.3s ease-in-out;
        box-shadow: 0px 6px 12px rgba(0, 173, 181, 0.3);
        display: block;
        width: 100%;
        text-align: center;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #008F96, #007B80);
        cursor: pointer;
        transform: translateY(-2px);  /* Lift effect on hover */
        box-shadow: 0px 8px 16px rgba(0, 173, 181, 0.5);
    }

    /* Input Fields */
    .stTextInput textarea, .stTextInput input {
        background-color: #2C2F33;
        border-radius: 10px;
        border: 2px solid #4D5159;
        padding: 14px;
        font-size: 16px;
        width: 100%;
        color: white;
        transition: border 0.3s ease, box-shadow 0.3s ease;
        outline: none;
    }

    .stTextInput textarea:focus, .stTextInput input:focus {
        border-color: #00ADB5;
        box-shadow: 0px 4px 10px rgba(0, 173, 181, 0.4);
    }

    .stTextInput label {
        font-size: 18px;
        font-weight: 600;
        color: #00ADB5;
    }

    /* Image Styling */
    .stImage {
        border-radius: 12px;
        box-shadow: 0px 6px 20px rgba(0, 0, 0, 0.3);
        display: block;
        margin: auto;
    }

    /* Markdown and Alerts */
    .stMarkdown {
        font-size: 18px;
        line-height: 1.7;
        text-align: justify;
    }

    .stAlert {
        font-size: 18px;
        background: #4D5159;
        color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.4);
        text-align: center;
    }

    /* Cards Styling */
    .stCard {
        background: #252B30;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        transform: translateZ(0);  /* Enable hardware acceleration for animation */
    }

    .stCard:hover {
        transform: scale(1.05);
        box-shadow: 0px 6px 20px rgba(0, 173, 181, 0.5);
    }

    /* Grid Layout for Cards */
    .stGrid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 30px;
        padding: 20px;
    }

    /* Footer */
    .footer {
        margin-top: 30px;
        text-align: center;
        font-size: 16px;
        color: #F0F0F0;
        padding: 15px;
        border-top: 1px solid #4D5159;
        opacity: 0.9;
    }

</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("C:\\Users\\Punith\\FINAL\\sentiment\\src\\1.jpeg", use_container_width=True)
    # st.title("Sentiment Analysis")
    st.markdown("Analyze sentiments in social media comments or reviews.")
    st.markdown("Get insights on how people feel about a topic or product using sentiment analysis.")
    st.markdown("---")
    #st.markdown("Developed by **Punith & Team**")

# Main App
st.title("💬 Social Media Sentiment Analysis")
st.subheader("Predict and visualize the sentiment of your text")

# Input Section
user_input = st.text_area("Enter your text below:", placeholder="Type something...", height=150, key="user_input")

# Loading spinner for prediction
with st.spinner("Analyzing sentiment..."):
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            # Clean and vectorize the input text
            cleaned_text = clean_text(user_input)
            text_vector = vectorizer.transform([cleaned_text])
            
            # Make sentiment prediction using Logistic Regression
            prediction = model.predict(text_vector)
            
            # Ensure case consistency and strip extra spaces
            predicted_label = prediction[0].strip().lower()

            # Sentiment label mapping
            sentiment_labels = {
                "negative": "😞 Negative", 
                "positive": "😊 Positive", 
                "neutral": "😐 Neutral"
            }

            # Get the corresponding sentiment
            sentiment = sentiment_labels.get(predicted_label, "Unknown")

            # Debugging: Show raw model output
            st.write(f"Raw model prediction output: '{prediction[0]}' (processed as '{predicted_label}')")

            # Display Sentiment Result
            if sentiment == "😊 Positive":
                st.success(f"**Predicted Sentiment:** {sentiment}")
            elif sentiment == "😞 Negative":
                st.error(f"**Predicted Sentiment:** {sentiment}")
            elif sentiment == "😐 Neutral":
                st.warning(f"**Predicted Sentiment:** {sentiment}")
            else:
                st.warning("**Predicted Sentiment:** Unknown. Please verify the model output.")

            
            # Mapping sentiment labels to numerical values
            score_mapping = {
                'positive': 1,
                'negative': -1,
                'neutral': 0,
                'irrelevant': 0
            }
            # Convert predicted sentiment to its corresponding numeric value
            sentiment_counts = {predicted_label: score_mapping[predicted_label]}

            # Display the updated bar chart
            st.pyplot(sentiment_scoring_lollipop_chart(sentiment_counts))

            
            # Optional: Sunburst Chart based on input text
            sunburst_chart = sunburst_visualization(cleaned_text)
            st.plotly_chart(sunburst_chart, use_container_width=True)


# Additional Features: Clear input button
if st.button("Clear Text"):     
    st.rerun()

# Footer
st.markdown("---")
#st.markdown("Made by **Punith & Team**")
