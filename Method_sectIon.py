import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Data Preparation
# Loading the CSV file
df = pd.read_csv('powertoys.csv')

# Ensure the Comment_Body is string
df['Comment_Body'] = df['Comment_Body'].astype(str)

# Step 2: Sentiment Analysis & Burnout Detection
# Function to classify sentiment
def classify_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Function to detect burnout based on keywords
def detect_burnout(text):
    burnout_keywords = [
        'exhausted', 'overwhelmed', 'stressed', 'stress', 'fatigue', 'burned out',
        'tired', 'frustrated', 'demotivated', 'collapse', 'depressed', 'drained',
        'curse', 'sad', 'worry', 'bad', 'frustate', 'frustation', 'mad'
    ]
    return 'Yes' if any(keyword in text.lower() for keyword in burnout_keywords) else 'No'

# Apply functions to DataFrame
df['sentiment_polarity'] = df['Comment_Body'].apply(classify_sentiment)
df['sentiment'] = df['sentiment_polarity'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')
df['burnout'] = df['Comment_Body'].apply(detect_burnout)

# Step 3: Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['Comment_Body'])
y = df['burnout'].values

# Encode the 'burnout' label
encoder = LabelEncoder()
encoded_y = encoder.fit_transform(y)

# Save the processed DataFrame to an Excel file
df.to_excel('processed_data.xlsx', index=False)
print("processed data is saved to: processed_data.xlsx")

# Visualizing nature of the Github comments
sentiment_counts = df['sentiment'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title('Sentiment Analysis Results')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Visualizing Burnout Detection Results
burnout_counts = df['burnout'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=burnout_counts.index, y=burnout_counts.values)
plt.title('Burnout Detection Results')
plt.xlabel('Burnout Detected')
plt.ylabel('Count')
plt.show()

# Model Building and Training
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X, encoded_y, df.index, test_size=0.2, random_state=42
)

model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train.toarray(), y_train, epochs=10, batch_size=32, validation_split=0.2)

# Model Evaluation
y_pred = model.predict(X_test.toarray()) > 0.5

Confusion Matrix Visualization
conf_mat = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='g', cmap='Blues',
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Creating a DataFrame with test indices, comments, actual and predicted labels
results_df = pd.DataFrame({
    'Comment_Body': df.loc[indices_test, 'Comment_Body'],
    'Actual_Burnout': encoder.inverse_transform(y_test),
    'Predicted_Burnout': encoder.inverse_transform(y_pred.flatten().astype(int))
}, index=indices_test)

# Save to Excel file
results_df.to_excel('burnout_predictions.xlsx', index=False)
print("The burnout data is saved to the file: burnout_predictions.xlsx")

# printing the classification report
print(classification_report(y_test, y_pred))
