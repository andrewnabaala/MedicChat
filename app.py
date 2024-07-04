import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load mapping.json
with open('mapping.json') as f:
    disease_mapping = json.load(f)

# Load symptom-disease datasets
train_data = pd.read_csv('symptom-disease-train-dataset.csv')
test_data = pd.read_csv('symptom-disease-test-dataset.csv')

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Preprocess the text data
train_data['text'] = train_data['text'].apply(preprocess_text)
test_data['text'] = test_data['text'].apply(preprocess_text)

# Combine train and test labels for fitting LabelEncoder
combined_labels = pd.concat([train_data['label'], test_data['label']])

# Encode the labels
label_encoder = LabelEncoder()
label_encoder.fit(combined_labels)

y_train = label_encoder.transform(train_data['label'])
y_test = label_encoder.transform(test_data['label'])

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_data['text'])
X_test = vectorizer.transform(test_data['text'])

# Define the model
xgb_model = XGBClassifier(use_label_encoder=True, eval_metric='mlogloss')

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [6, 10],
    'learning_rate': [0.01, 0.1]
}

# Perform grid search
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Function to predict disease based on symptoms
def predict_disease(symptoms):
    symptoms = preprocess_text(symptoms)
    symptoms_vector = vectorizer.transform([symptoms])
    prediction = best_model.predict(symptoms_vector)
    disease_label = label_encoder.inverse_transform(prediction)[0]

    disease_name = next(key for key, value in disease_mapping.items() if value == disease_label)
    return disease_name

# Interactive Chatbot
def chatbot():
    print("Hello! What's your name?")
    name = input("Your name: ")

    print(f"Hi {name}! I'm your medical assistant chatbot. You can tell me your symptoms, and I'll try to predict the disease.")
    print("Type 'bye' or 'exit' to end the conversation.")
    while True:
        user_input = input("Enter your symptoms: ")
        if user_input.lower() in ['bye', 'exit']:
            print(f"Goodbye, {name}! Take care.")
            break
        predicted_disease = predict_disease(user_input)
        print(f"{name}, based on your symptoms, the predicted disease is: {predicted_disease}")
chatbot()