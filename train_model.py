import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# Simulate the dataset
data = {
    'message': ["Hey, how are you?", "WINNER!! Click to claim your prize", "Can we meet tomorrow?", "You have won a $1000 gift card!"],
    'label': ['ham', 'spam', 'ham', 'spam'],
    'is_number_saved': [1, 0, 1, 0],  # 1 if the number is saved, 0 otherwise
    'message_type': ['sms', 'sms', 'email', 'email']  # 'sms' or 'email'
}

df = pd.DataFrame(data)

# Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X = df[['message', 'is_number_saved', 'message_type']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data and create pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('text', CountVectorizer(), 'message'),
        ('cat', OneHotEncoder(), ['is_number_saved', 'message_type'])
    ])

model = make_pipeline(preprocessor, MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'spam_detector.pkl')
