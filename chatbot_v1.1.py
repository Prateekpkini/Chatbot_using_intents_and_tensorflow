import os
import json
import datetime
import csv
import nltk
import ssl
import shutil
import streamlit as st
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from nltk.stem import WordNetLemmatizer

# =============================================
# NLTK INITIALIZATION WITH ROBUST ERROR HANDLING
# =============================================

def initialize_nltk():
    try:
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        possible_nltk_paths = [
            os.path.join(os.path.expanduser("~"), "nltk_data"),
            os.path.join(os.getcwd(), "nltk_data"),
            os.path.join(os.environ.get('APPDATA', ''), "nltk_data"),
            "C:/nltk_data",
            "D:/nltk_data"
        ]

        nltk.data.path.clear()
        for path in possible_nltk_paths:
            try:
                os.makedirs(path, exist_ok=True)
                nltk.data.path.append(path)
            except Exception as e:
                print(f"Couldn't setup NLTK path {path}: {e}")

        required_resources = {
            'punkt': ['tokenizers/punkt'],
            'wordnet': ['corpora/wordnet'],
            'omw-1.4': ['corpora/omw']
        }

        for resource, subpaths in required_resources.items():
            try:
                nltk.download(resource, download_dir=nltk.data.path[0])
                for subpath in subpaths:
                    full_path = os.path.join(nltk.data.path[0], subpath)
                    if not os.path.exists(full_path):
                        raise FileNotFoundError(f"Subpath {subpath} not found after download")
            except Exception as e:
                print(f"Failed to download {resource}: {e}")

        try:
            tokens = nltk.word_tokenize("This is a test sentence.")
        except Exception:
            nltk.word_tokenize = lambda text: text.lower().split()

    except Exception as e:
        raise SystemExit("Failed to initialize NLTK")

initialize_nltk()

# =============================================
# CORE CHATBOT FUNCTIONALITY
# =============================================

class ChatbotEngine:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = self.load_intents()
        self.data = self.prepare_data()
        self.tf_model, self.lr_model = self.train_models()

    def load_intents(self):
        file_path = os.path.abspath("./intents.json")
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'intents' in data:
                    return data['intents']
                return data
        except Exception:
            return []

    def preprocess_text(self, text):
        try:
            tokens = nltk.word_tokenize(text.lower())
        except Exception:
            tokens = text.lower().split()
        return ' '.join([self.lemmatizer.lemmatize(token) for token in tokens])

    def prepare_data(self):
        tags, patterns = [], []
        for intent in self.intents:
            if 'tag' in intent and 'patterns' in intent:
                tags.extend([intent['tag']] * len(intent['patterns']))
                patterns.extend(intent['patterns'])
        data = pd.DataFrame({'patterns': patterns, 'tags': tags})
        data['processed'] = data['patterns'].apply(self.preprocess_text)
        return data

    def train_tensorflow_model(self, train_x, train_y):
        model = Sequential([
            Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(train_y[0]), activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='loss', patience=5)
        model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1, callbacks=[early_stop])
        return model

    def train_logistic_regression(self, x, y):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        clf = LogisticRegression(random_state=0, max_iter=10000)
        clf.fit(x, y_encoded)
        return clf, le

    def train_models(self):
        words, classes, documents = [], [], []
        for intent in self.intents:
            if 'tag' in intent and 'patterns' in intent:
                for pattern in intent['patterns']:
                    tokens = nltk.word_tokenize(pattern)
                    words.extend(tokens)
                    documents.append((tokens, intent['tag']))
                    if intent['tag'] not in classes:
                        classes.append(intent['tag'])

        ignore_words = ['?', '!', '.', ',']
        words = sorted(list(set([self.lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words])))
        classes = sorted(list(set(classes)))

        training = []
        output_empty = [0] * len(classes)
        for doc in documents:
            bag = [0] * len(words)
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in doc[0]]
            for i, w in enumerate(words):
                if w in pattern_words:
                    bag[i] = 1
            output_row = output_empty[:]
            output_row[classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training, dtype=object)
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        tf_model = self.train_tensorflow_model(train_x, train_y)

        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(self.data['processed'])
        lr_model, le = self.train_logistic_regression(x, self.data['tags'])

        return (tf_model, words, classes), (lr_model, le, vectorizer)

    def tf_predict(self, input_text):
        processed_input = self.preprocess_text(input_text)
        tokens = nltk.word_tokenize(processed_input)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        bag = [0] * len(self.tf_model[1])
        for s in tokens:
            for i, w in enumerate(self.tf_model[1]):
                if w == s:
                    bag[i] = 1
        results = self.tf_model[0].predict(np.array([bag]))[0]
        results_index = np.argmax(results)
        tag = self.tf_model[2][results_index]
        for intent in self.intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses']), results[results_index]
        return "I'm not sure how to respond to that.", 0.0

    def lr_predict(self, input_text):
        processed_input = self.preprocess_text(input_text)
        input_vec = self.lr_model[2].transform([processed_input])
        tag_encoded = self.lr_model[0].predict(input_vec)[0]
        tag = self.lr_model[1].inverse_transform([tag_encoded])[0]
        for intent in self.intents:
            if intent['tag'] == tag:
                proba = self.lr_model[0].predict_proba(input_vec)[0][tag_encoded]
                return random.choice(intent['responses']), proba
        return "I'm not sure how to respond to that.", 0.0

# =============================================
# STREAMLIT UI
# =============================================

def main():
    st.title("Enhanced Chatbot with TensorFlow")

    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            st.session_state.chatbot = ChatbotEngine()

    model_choice = st.sidebar.radio("Select Model", ("TensorFlow Neural Network", "Logistic Regression"), index=0)
    menu = ["Home", "Conversation History", "About", "Model Performance"]
    choice = st.sidebar.selectbox("Menu", menu)

    if not os.path.exists('chat_log.csv'):
        with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['User Input', 'Chatbot Response', 'Model Used', 'Confidence', 'Timestamp'])

    if choice == "Home":
        st.write("Welcome to the chatbot. Type a message and press Enter.")
        st.info(f"Currently using: {model_choice}")
        user_input = st.text_input("You:", key="user_input")
        if user_input:
            try:
                if model_choice == "TensorFlow Neural Network":
                    response, confidence = st.session_state.chatbot.tf_predict(user_input)
                else:
                    response, confidence = st.session_state.chatbot.lr_predict(user_input)
                st.text_area("Chatbot:", value=response, height=120)
                st.write(f"Confidence: {confidence:.2%}")
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([user_input, response, model_choice, confidence, timestamp])
            except Exception as e:
                st.error(f"Error: {e}")

    elif choice == "Conversation History":
        st.header("Conversation History")
        try:
            history = pd.read_csv('chat_log.csv')
            if not history.empty:
                st.dataframe(history)
            else:
                st.warning("No conversation history yet.")
        except Exception as e:
            st.error(f"Couldn't load history: {e}")

    elif choice == "Model Performance":
        st.header("Model Performance")
        st.subheader("TensorFlow Neural Network")
        st.write("""
        - 3-layer neural network (128-64-output)
        - ReLU activations, Softmax output
        - Dropout (0.5), Adam optimizer
        - Early stopping with patience=5
        """)
        st.subheader("Logistic Regression")
        st.write("""
        - TF-IDF vectorization
        - L2 regularization
        - Max iterations: 10,000
        """)

    elif choice == "About":
        st.header("About")
        st.write("""
        This chatbot supports two models: a TensorFlow/Keras neural network and a Logistic Regression model.

        Features:
        - Text preprocessing (lemmatization)
        - Confidence scoring
        - Chat logging
        - Streamlit interface
        """)

if __name__ == '__main__':
    main()