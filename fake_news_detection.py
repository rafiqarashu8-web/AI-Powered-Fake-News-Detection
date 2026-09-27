import pandas as pd
import numpy as np
import os
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ============================
# 1. LOAD DATASET
# ============================
path = r"C:\Users\user\Documents\rafiqa\dataset\train.csv"

if not os.path.exists(path):
    print("Download dataset first from Kaggle: 'Fake and Real News Dataset'")
    exit()

df = pd.read_csv(path)
print(f"Dataset Loaded: {df.shape}")

# If dataset has separate Fake.csv and True.csv, combine them first
# df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# ============================
# 2. DATA CLEANING & PREPROCESSING
# ============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text) # remove []
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # remove links
    text = re.sub(r'<.*?>+', '', text) # remove html tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # remove punctuation
    text = re.sub(r'\n', '', text) # remove new lines
    text = re.sub(r'\w*\d\w*', '', text) # remove words with numbers
    return text

df['content'] = df['title'] + " " + df['text']
df['content'] = df['content'].apply(clean_text)
df = df.dropna()

print("Data cleaned. Sample:")
print(df['content'].head())

# ============================
# 3. FEATURE EXTRACTION - TF-IDF
# ============================
tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
X = tfidf.fit_transform(df['content'])
y = df['label']

# ============================
# 4. TRAIN TEST SPLIT
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ============================
# 5. TRAIN MULTIPLE MODELS
# ============================
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n--- {name} ---")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

# ============================
# 6. PLOT RESULTS
# ============================
plt.figure(figsize=(8,5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.0)
plt.savefig("accuracy_comparison.png")
plt.show()

# ============================
# 7. SAVE BEST MODEL
# ============================
best_model = models["Decision Tree"] # Change if another is better
import joblib
joblib.dump(best_model, "fake_news_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
print("\nModel saved successfully!")

# ============================
# 8. TEST WITH NEW INPUT
# ============================
def predict_news(news):
    news_clean = clean_text(news)
    news_tfidf = tfidf.transform([news_clean])
    pred = best_model.predict(news_tfidf)
    prob = best_model.predict_proba(news_tfidf)
    if pred[0] == 1:
        return f"REAL NEWS | Confidence: {prob[0][1]*100:.2f}%"
    else:
        return f"FAKE NEWS | Confidence: {prob[0][0]*100:.2f}%"

print("\nTest:", predict_news("Government launches free education for all students"))

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("Accuracy:", accuracy_score(y_test, y_pred)*100, "%")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
