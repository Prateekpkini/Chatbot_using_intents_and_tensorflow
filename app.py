import os
import json
import datetime
import csv
import nltk
import ssl
import shutil

# =============================================
# NLTK INITIALIZATION WITH ROBUST ERROR HANDLING
# =============================================

def initialize_nltk():
    """Set up NLTK with multiple fallback options and verification"""
    try:
        # Configure SSL context
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Set up multiple possible NLTK data paths
        possible_nltk_paths = [
            os.path.join(os.path.expanduser("~"), "nltk_data"),
            os.path.join(os.getcwd(), "nltk_data"),
            os.path.join(os.environ.get('APPDATA', ''), "nltk_data"),
            "C:/nltk_data",
            "D:/nltk_data"
        ]

        # Clear existing paths and add our preferred ones
        nltk.data.path.clear()
        for path in possible_nltk_paths:
            try:
                os.makedirs(path, exist_ok=True)
                nltk.data.path.append(path)
            except Exception as e:
                print(f"Couldn't setup NLTK path {path}: {e}")

        # Download required resources with verification
        required_resources = {
            'punkt': ['tokenizers/punkt', 'tokenizers/punkt_tab'],
            'wordnet': ['corpora/wordnet'],
            'omw-1.4': ['corpora/omw']
        }

        for resource, subpaths in required_resources.items():
            try:
                print(f"Attempting to download {resource}...")
                nltk.download(resource, download_dir=nltk.data.path[0])
                
                # Verify the files actually exist
                for subpath in subpaths:
                    full_path = os.path.join(nltk.data.path[0], subpath)
                    if not os.path.exists(full_path):
                        raise FileNotFoundError(f"Subpath {subpath} not found after download")
                
                print(f"Successfully installed {resource}")
            except Exception as e:
                print(f"Failed to download {resource}: {e}")
                if resource == 'punkt':
                    print("Warning: punkt is essential for tokenization - some functionality may be limited")

        # Test tokenization
        test_text = "This is a test sentence."
        try:
            tokens = nltk.word_tokenize(test_text)
            print("NLTK tokenizer test successful:", tokens)
        except Exception as e:
            print("Tokenization test failed, implementing fallback:", e)
            # Implement simple fallback tokenizer
            nltk.word_tokenize = lambda text: text.lower().split()
            print("Using fallback whitespace tokenizer")

    except Exception as e:
        print(f"Critical error during NLTK initialization: {e}")
        raise SystemExit("Failed to initialize NLTK - exiting")

# Initialize NLTK before any other imports
initialize_nltk()

# =============================================
# MAIN IMPORTS (AFTER NLTK IS INITIALIZED)
# =============================================
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
# CORE CHATBOT FUNCTIONALITY
# =============================================

class ChatbotEngine:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = self.load_intents()
        self.data = self.prepare_data()
        self.tf_model, self.lr_model = self.train_models()
        
    def load_intents(self):
        """Load intents from JSON file with flexible structure handling"""
        file_path = os.path.abspath("./intents.json")
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                
                # Handle both formats:
                # 1. Direct list of intents
                # 2. Dictionary with 'intents' key
                if isinstance(data, list):
                    return data  # Return the list directly
                elif isinstance(data, dict) and 'intents' in data:
                    return data['intents']  # Return just the intents list
                return data  # Fallback return whatever we got
                
        except Exception as e:
            st.error(f"Failed to load intents: {e}")
            return []  # Return empty list instead of dict

    def preprocess_text(self, text):
        """Robust text preprocessing with fallbacks"""
        try:
            # Tokenize with NLTK or fallback
            tokens = nltk.word_tokenize(text.lower())
        except Exception as e:
            print(f"Tokenization failed, using fallback: {e}")
            tokens = text.lower().split()
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)

    def prepare_data(self):
        """Prepare training data handling both list and dict formats"""
        tags, patterns = [], []
        
        # Ensure we're working with a list of intents
        intents_list = self.intents if isinstance(self.intents, list) else []
        
        for intent in intents_list:
            if 'tag' in intent and 'patterns' in intent:
                tags.extend([intent['tag']] * len(intent['patterns']))
                patterns.extend(intent['patterns'])
        
        data = pd.DataFrame({'patterns': patterns, 'tags': tags})
        if not data.empty:
            data['processed'] = data['patterns'].apply(self.preprocess_text)
        return data

    def train_tensorflow_model(self, train_x, train_y):
        """Build and train TensorFlow model"""
        model = Sequential([
            Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(train_y[0]), activation='softmax')
        ])
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.01),
            metrics=['accuracy']
        )
        
        early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
        model.fit(
            np.array(train_x), 
            np.array(train_y), 
            epochs=200, 
            batch_size=5, 
            verbose=1, 
            callbacks=[early_stop]
        )
        return model

    def train_logistic_regression(self, x, y):
        """Train Logistic Regression model"""
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        clf = LogisticRegression(random_state=0, max_iter=10000)
        clf.fit(x, y_encoded)
        return clf, le

    def train_models(self):
        """Train both models with proper data preparation"""
        # Prepare TF model data
        words, classes, documents = [], [], []
        ignore_words = ['?', '!', '.', ',']
        
        # Ensure we're working with a list of intents
        intents_list = self.intents if isinstance(self.intents, list) else []
        
        for intent in intents_list:
            if 'tag' in intent and 'patterns' in intent:
                for pattern in intent['patterns']:
                    words.extend(nltk.word_tokenize(pattern))
                    documents.append((nltk.word_tokenize(pattern), intent['tag']))
                    if intent['tag'] not in classes:
                        classes.append(intent['tag'])
        
        words = [self.lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
        words = sorted(list(set(words)))
        classes = sorted(list(set(classes)))
        
        # Create training data
        training = []
        output_empty = [0] * len(classes)
        
        for doc in documents:
            bag = [0] * len(words)
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in doc[0]]
            
            for i, w in enumerate(words):
                bag[i] = 1 if w in pattern_words else 0
            
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            training.append([bag, output_row])
        
        random.shuffle(training)
        training = np.array(training, dtype=object)
        
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        
        # Train TensorFlow model
        tf_model = self.train_tensorflow_model(train_x, train_y)
        
        # Train Logistic Regression
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(self.data['processed'])
        lr_model, le = self.train_logistic_regression(x, self.data['tags'])
        
        return (tf_model, words, classes), (lr_model, le, vectorizer)

    def tf_predict(self, input_text):
        """Make prediction using TensorFlow model"""
        processed_input = self.preprocess_text(input_text)
        tokens = nltk.word_tokenize(processed_input)
        tokens = [self.lemmatizer.lemmatize(token.lower()) for token in tokens]
        
        # Create bag of words
        bag = [0] * len(self.tf_model[1])  # words list
        for s in tokens:
            for i, w in enumerate(self.tf_model[1]):
                if w == s:
                    bag[i] = 1
                    
        # Predict
        results = self.tf_model[0].predict(np.array([bag]))[0]
        results_index = np.argmax(results)
        tag = self.tf_model[2][results_index]  # classes list
        
        # Get response
        intents_list = self.intents if isinstance(self.intents, list) else []
        for intent in intents_list:
            if 'tag' in intent and intent['tag'] == tag:
                return random.choice(intent['responses']), results[results_index]
        return "I'm not sure how to respond to that.", 0.0

    def lr_predict(self, input_text):
        """Make prediction using Logistic Regression"""
        processed_input = self.preprocess_text(input_text)
        input_vec = self.lr_model[2].transform([processed_input])  # vectorizer
        tag_encoded = self.lr_model[0].predict(input_vec)[0]  # lr_model
        tag = self.lr_model[1].inverse_transform([tag_encoded])[0]  # le
        
        intents_list = self.intents if isinstance(self.intents, list) else []
        for intent in intents_list:
            if 'tag' in intent and intent['tag'] == tag:
                proba = self.lr_model[0].predict_proba(input_vec)[0][tag_encoded]
                return random.choice(intent['responses']), proba
        return "I'm not sure how to respond to that.", 0.0

# =============================================
# STREAMLIT UI
# =============================================

def main():
    st.title("Enhanced Chatbot with TensorFlow")
    
    # Initialize chatbot engine
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            st.session_state.chatbot = ChatbotEngine()
    
    # Model selection
    model_choice = st.sidebar.radio(
        "Select Model",
        ("TensorFlow Neural Network", "Logistic Regression"),
        index=0
    )
    
    # Menu options
    menu = ["Home", "Conversation History", "About", "Model Performance"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Initialize chat log
    if not os.path.exists('chat_log.csv'):
        with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['User Input', 'Chatbot Response', 'Model Used', 'Confidence', 'Timestamp'])

    if choice == "Home":
        st.write("Welcome to the enhanced chatbot. Please type a message and press Enter to start the conversation.")
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
                
                # Log conversation
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([user_input, response, model_choice, confidence, timestamp])
                
                if response.lower() in ['goodbye', 'bye']:
                    st.success("Thank you for chatting with me. Have a great day!")
            except Exception as e:
                st.error(f"Error generating response: {e}")
                st.error("Please try a different input or check the console for details")

    elif choice == "Conversation History":
        st.header("Conversation History")
        try:
            history = pd.read_csv('chat_log.csv')
            if not history.empty:
                st.dataframe(history)
            else:
                st.warning("No conversation history available yet.")
        except Exception as e:
            st.error(f"Couldn't load history: {e}")

    elif choice == "Model Performance":
        st.header("Model Performance Comparison")
        st.subheader("TensorFlow Neural Network")
        st.write("""
        - Architecture: 3-layer neural network (128-64-output)
        - Activation: ReLU for hidden layers, Softmax for output
        - Dropout: 0.5 for regularization
        - Optimizer: Adam with learning rate 0.01
        - Training: Early stopping with patience=5
        """)
        
        st.subheader("Logistic Regression")
        st.write("""
        - Uses TF-IDF vectorization
        - Regularization: L2 by default
        - Max iterations: 10,000
        """)
        
        st.write("Note: The TensorFlow model typically provides better performance for complex patterns but requires more computational resources.")

    elif choice == "About":
        st.write("""
        ## Enhanced Chatbot with TensorFlow and Logistic Regression
        
        This chatbot features two different machine learning models for intent classification:
        1. A TensorFlow/Keras neural network
        2. A traditional Logistic Regression model with TF-IDF features
        
        Key improvements:
        - Added text preprocessing with lemmatization
        - Better tokenization and feature engineering
        - Confidence scoring for responses
        - Model comparison capability
        - Enhanced conversation history tracking
        """)

        st.subheader("Technical Details")
        st.write("""
        - **Natural Language Processing**: NLTK for tokenization, WordNetLemmatizer for text normalization
        - **Machine Learning**: Two different approaches for comparison
        - **User Interface**: Streamlit with sidebar navigation
        - **Data Persistence**: CSV logging of all conversations
        """)

if __name__ == '__main__':
    main()