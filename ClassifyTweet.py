
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from textblob import TextBlob  # Simple sentiment analysis tool

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------- Load and Combine Datasets -------- #
print("\nLoading datasets...")
df1 = pd.read_csv("tweets/data_analysis.csv", low_memory=False)
df2 = pd.read_csv("tweets/data_science.csv", low_memory=False)
df3 = pd.read_csv("tweets/data_visualization.csv", low_memory=False)

df = pd.concat([df1, df2, df3], ignore_index=True)
print("✅ Datasets loaded and combined. Total rows:", df.shape[0])

# -------- Inspect Columns -------- #
print("\nAvailable columns:", df.columns.tolist())

# Rename 'tweet' to 'text' if available
if 'tweet' in df.columns:
    df.rename(columns={'tweet': 'text'}, inplace=True)
    print("✅ Renamed 'tweet' column to 'text'.")

# -------- Assign Sentiment Labels -------- #
print("\nAssigning sentiment using TextBlob...")
df['sentiment'] = df['text'].apply(lambda x: 'positive' if TextBlob(str(x)).sentiment.polarity > 0 else 'negative')
print("✅ Sentiment assigned based on polarity.")

# Show sample rows
print("\nSample labeled data:")
print(df[['text', 'sentiment']].head(5))

# -------- Clean and Filter Data -------- #
print("\nCleaning text...")
df = df[['text', 'sentiment']].dropna()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)
print("✅ Text cleaned.")

# Show cleaned sample
print("\nSample cleaned text:")
print(df[['clean_text']].head(5))

# -------- Encode Labels and Split -------- #
print("\nEncoding sentiment labels...")
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
print("✅ Labels encoded: 1 for Positive, 0 for Negative")

print("\nSentiment label distribution:")
print(df['label'].value_counts())

print("\nSplitting data into train and test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)
print(f"✅ Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# -------- Vectorization -------- #
print("\nVectorizing text...")
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("✅ Text vectorized. Feature count:", X_train_vec.shape[1])

# -------- Train Model -------- #
print("\nTraining Logistic Regression model...")
model = LogisticRegression()
model.fit(X_train_vec, y_train)
print("✅ Model training complete.")

# -------- Evaluate Model -------- #
print("\nPredicting on test data...")
y_pred = model.predict(X_test_vec)

print("\n--- Model Evaluation ---")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------- Confusion Matrix -------- #
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("✅ Done. Confusion matrix saved as 'confusion_matrix.png'")
