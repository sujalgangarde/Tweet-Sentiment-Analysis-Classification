import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from textblob import TextBlob  # Simple sentiment analysis tool

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------- Streamlit UI Setup -------- #
st.title("Tweet Sentiment Classifier")
st.write("This app performs sentiment analysis on tweets.")

# -------- Load and Combine Datasets -------- #
st.write("Loading datasets...")
df1 = pd.read_csv("tweets/data_analysis.csv", low_memory=False)
df2 = pd.read_csv("tweets/data_science.csv", low_memory=False)
df3 = pd.read_csv("tweets/data_visualization.csv", low_memory=False)

# Combine all datasets
df = pd.concat([df1, df2, df3], ignore_index=True)
st.write("Datasets loaded and combined.")
st.write(f"Total number of records: {df.shape[0]}")

# -------- Inspect Columns -------- #
st.write("Available columns:", df.columns.tolist())

# Rename 'tweet' to 'text' for consistency
df.rename(columns={'tweet': 'text'}, inplace=True)

# -------- Assign Sentiment Labels -------- #
# Use TextBlob for simple sentiment analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    return 'positive' if analysis.sentiment.polarity > 0 else 'negative'

df['sentiment'] = df['text'].apply(get_sentiment)

st.write(f"Sentiment labels assigned. Sample data:")
st.write(df[['text', 'sentiment']].head())

# -------- Clean and Filter Data -------- #
st.write("Cleaning and filtering data...")
df = df[['text', 'sentiment']].dropna()

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

st.write("Data cleaned. Sample cleaned text:")
st.write(df[['text', 'clean_text']].head())

# -------- Encode Labels and Split -------- #
st.write("Encoding and splitting data...")
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------- Train Model -------- #
st.write("Training model...")
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# -------- Evaluate Model -------- #
st.write("Evaluating model...")
y_pred = model.predict(X_test_vec)

st.write("\n--- Model Evaluation ---")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# -------- Plot Confusion Matrix -------- #
st.write("Plotting confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()

# Show the plot in Streamlit
st.pyplot(plt)

# Save the plot as an image
plt.savefig("confusion_matrix.png")
st.write("Confusion matrix saved as 'confusion_matrix.png'.")

st.write("Done.")

