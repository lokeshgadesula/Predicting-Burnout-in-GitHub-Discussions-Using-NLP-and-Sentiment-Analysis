import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Data Preparation
df = pd.read_csv('csvfile.csv')

df['Comment_Body'] = df['Comment_Body'].astype(str)

# Sentiment Analysis & Burnout Detection
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

df['sentiment_polarity'] = df['Comment_Body'].apply(classify_sentiment)
df['sentiment'] = df['sentiment_polarity'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')
df['burnout'] = df['Comment_Body'].apply(detect_burnout)

# Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['Comment_Body'])
y = df['burnout'].values

encoder = LabelEncoder()
encoded_y = encoder.fit_transform(y)

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

# Step 5: Model Evaluation
y_pred = model.predict(X_test.toarray()) > 0.5

# Creating a DataFrame with test indices, comments, actual and predicted labels
results_df = pd.DataFrame({
    'Comment_Body': df.loc[indices_test, 'Comment_Body'],
    'Actual_Burnout': encoder.inverse_transform(y_test),
    'Predicted_Burnout': encoder.inverse_transform(y_pred.flatten().astype(int))
}, index=indices_test)

try:
    results_df.to_excel('Burnout_Predictions.xlsx', index=False)
    print(f'The excel file was successfully saved to "Burnout_Predictions.xlsx"')
except Exception as e:
    print(f'An error occurred while saving the excel file: {e}')

