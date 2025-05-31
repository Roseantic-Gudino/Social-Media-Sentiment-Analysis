import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
from io import BytesIO

def sentiment_scoring_lollipop_chart(sentiment_scores):
    """Generates a modern lollipop chart displaying sentiment scores (-1, 0, 1)."""

    sns.set_theme(style="whitegrid")  # Ensure a white background

    # Sentiment to numeric mapping
    score_mapping = {
        'positive': 1,
        'negative': -1,
        'neutral': 0,
        'irrelevant': 0
    }

    # Convert sentiment scores to a DataFrame
    df = pd.DataFrame({'Sentiment': list(sentiment_scores.keys()), 'Score': list(sentiment_scores.values())})
    df['Mapped Score'] = df['Sentiment'].map(score_mapping)  # Apply mapping

    # Ensure correct order of categories
    sentiment_order = ['positive', 'neutral', 'negative', 'irrelevant']
    df['Sentiment'] = pd.Categorical(df['Sentiment'], categories=sentiment_order, ordered=True)
    df = df.sort_values(by='Sentiment')

    # Color mapping for better contrast
    color_map = {
        'positive': '#1f77b4',  # Dark Blue
        'negative': '#d62728',  # Dark Red
        'neutral': '#2ca02c',   # Dark Green
        'irrelevant': '#ff7f0e' # Orange
    }

    # Extract values
    labels = df['Sentiment'].tolist()
    values = df['Mapped Score'].tolist()
    colors = [color_map.get(sent, '#d3d3d3') for sent in labels]

    # Figure and Axes
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    
    # Explicitly set white background
    fig.patch.set_facecolor('white')  # Set figure background
    ax.set_facecolor('white')         # Set axis background

    # Create lollipop sticks
    ax.vlines(x=labels, ymin=0, ymax=values, color=colors, linewidth=2, linestyle='dotted')

    # Create lollipop circles
    ax.scatter(labels, values, color=colors, s=200, edgecolors='black', linewidth=1.2, zorder=3)

    # Display values on top of points
    for i, (label, value) in enumerate(zip(labels, values)):
        ax.text(label, value + (0.05 if value >= 0 else -0.1), f"{int(value)}", 
                ha='center', va='bottom' if value >= 0 else 'top', 
                fontsize=12, fontweight='bold', color='black')

    # Customizing axes and labels
    ax.set_ylabel('Sentiment Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sentiment Category', fontsize=14, fontweight='bold')
    # ax.set_title('Sentiment Confidence Scores (Lollipop Chart)', fontsize=16, fontweight='bold')

    # Improve grid appearance
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # Adjust y-axis range for better spacing
    ax.set_ylim(-1.2, 1.2)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig



def sunburst_visualization(text):
    words = text.split() 
    word_counts = pd.DataFrame(words, columns=["Word"]).value_counts().reset_index()
    word_counts.columns = ["Word", "Frequency"]
    fig = px.sunburst(word_counts, path=["Word"], values="Frequency", color="Frequency",
    color_continuous_scale="Blues", title="Word Frequency Sunburst Chart")
    return fig
