import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from joblib import dump, load

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.use('Agg')

import graphs  

vectorizer = TfidfVectorizer()  


def calculate_entropy(password):
    """
    Calculates the entropy of a given password based on the frequency of each character.
    Entropy is a measure of the randomness or information content of the password.

    Args:
        password (str): The password for which entropy is to be calculated.

    Returns:
        float: The entropy value of the password.
    """
    characters = list(password)
   
    char_counts = {}
    for char in characters:
        if char in char_counts:
            char_counts[char] += 1
        else:
            char_counts[char] = 1
           
    total_chars = len(characters)
   
    probabilities = [count / total_chars for count in char_counts.values()]
   
    # Calculate the entropy using the formula: E = -sum(p * log2(p)) for all p
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)  #  p>0to avoid log2(0)
   
    return entropy


def transform_password(password):
    """
    Transforms a password into a feature set suitable for machine learning models.
    This involves calculating several features such as password length, number of uppercase letters,
    number of lowercase letters, digits, special characters, and entropy.

    Args:
        password (str): The password to transform.

    Returns:
        sparse matrix: A combined sparse matrix of TF-IDF features and additional engineered features.
    """
    features = {
        'length': len(password),
        'num_uppercase': sum(1 for c in password if c.isupper()),
        'num_lowercase': sum(1 for c in password if c.islower()),
        'num_digits': sum(1 for c in password if c.isdigit()),
        'num_special': sum(1 for c in password if not c.isalnum()),
        'entropy': calculate_entropy(password)
    }
    features_df = pd.DataFrame([features])
    vectorizer = load('./model/tfidf_vectorizer.joblib')
    features_tfidf = vectorizer.transform([password])
    combined_features = hstack([features_tfidf, features_df.astype(float)])

    return combined_features


def train_model():
    """
    Trains a machine learning model to predict password strength.
    This function reads the password data, extracts features, combines them with TF-IDF vectorized text data,
    and then uses a RandomForestClassifier to train the model.

    Saves the trained model and vectorizer for future use.

    Outputs:
        Displays a confusion matrix and prints the accuracy of the model on the test data.
    """
    data = graphs.read_data()

    # data = data.sample(n=30000, random_state=42)

    data['length'] = data['password'].apply(len)
    data['num_uppercase'] = data['password'].apply(lambda x: sum(1 for c in x if c.isupper()))
    data['num_lowercase'] = data['password'].apply(lambda x: sum(1 for c in x if c.islower()))
    data['num_digits'] = data['password'].apply(lambda x: sum(1 for c in x if c.isdigit()))
    data['num_special'] = data['password'].apply(lambda x: sum(1 for c in x if not c.isalnum()))
    data['entropy'] = data['password'].apply(calculate_entropy)

    # Convert passwords to a matrix of TF-IDF features
    X_tfidf = vectorizer.fit_transform(data['password'])
    features_df = pd.DataFrame(data[['length', 'num_uppercase', 'num_lowercase', 'num_digits', 'num_special', 'entropy']].values,
                               columns=['length', 'num_uppercase', 'num_lowercase', 'num_digits', 'num_special', 'entropy'])

    # Combine the TF-IDF features and the engineered features
    X = hstack([X_tfidf, features_df.astype(float)])
    y = data['strength']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    classifier.fit(X_train, y_train)

    # save trained model & vectorizer
    dump(classifier, 'password_strength_classifier.joblib')
    dump(vectorizer, 'tfidf_vectorizer.joblib')

    y_pred = classifier.predict(X_test)

    # Make confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy) # current accuracy of test data is 100%


def predict_strength(password):
    """
    Predicts the strength of a given password using a pre-trained model.

    Args:
        password (str): The password to predict strength for.

    Returns:
        int: The predicted strength level of the password (0 = weak, 1 = medium, 2 = strong).
    """
    classifier = load('./model/password_strength_classifier.joblib')
    features = transform_password(password)
    predicted_strength = classifier.predict(features)
    # print("Predicted Strength: ", predicted_strength[0])
    return predicted_strength[0]

# testing
# train_model()  
# predict_strength("examplePassword123") 
