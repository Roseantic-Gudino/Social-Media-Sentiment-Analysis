import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preserve important words
negation_words = {"no", "not", "never", "none", "nor", "n't"}
essential_words = {"is", "was", "am", "are", "be", "being", "been", "colourful", "beautiful", "happy", "love", "great"}
contrast_words = {"but", "however", "although", "yet", "though"}

stop_words -= negation_words  # Keep negation words
stop_words -= essential_words  # Keep words influencing sentiment

# Initialize VADER for sentiment analysis
vader = SentimentIntensityAnalyzer()

def clean_text(text):
    """Preprocesses text while preserving sentiment context."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9.!?']", " ", text)  # Keep key punctuation

    tokens = word_tokenize(text)
    processed_tokens = []
    contains_contrast = False
    negation_active = False

    for word in tokens:
        if word in contrast_words:
            contains_contrast = True  # Mark contrast for sarcasm detection
        elif word in negation_words:
            negation_active = True  # Mark negation start
        elif word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            if negation_active:  
                lemma = "not_" + lemma  # Preserve negation context
                negation_active = False  # Reset negation flag
            processed_tokens.append(lemma)

    if contains_contrast:
        processed_tokens.append("mixed_sentiment")

    return ' '.join(processed_tokens)

def get_sentiment(text):
    """Combines VADER & TextBlob for more accurate sentiment analysis."""
    cleaned_text = clean_text(text)
    
    # VADER sentiment
    vader_score = vader.polarity_scores(cleaned_text)
    compound_score = vader_score["compound"]

    # TextBlob sentiment
    blob_score = TextBlob(cleaned_text).sentiment.polarity

    # Ensemble Decision
    final_score = (compound_score + blob_score) / 2  # Average both models

    # Handle extreme cases (e.g., "not bad" should be positive)
    if "not" in cleaned_text and any(pos_word in cleaned_text for pos_word in essential_words):
        final_score += 0.3  # Boost positivity in "not bad" cases

    # Classify sentiment
    if final_score > 0.2:
        sentiment = "Positive 😊"
    elif final_score < -0.2:
        sentiment = "Negative 😡"
    else:
        sentiment = "Neutral 😐"
    
    print(f"\n📝 Original: {text}")
    print(f"🔍 Processed: {cleaned_text}")
    print(f"📊 VADER Score: {compound_score}, TextBlob Score: {blob_score}")
    print(f"🔎 Final Score: {final_score}")
    print(f"✅ Sentiment: {sentiment}")

    return sentiment


