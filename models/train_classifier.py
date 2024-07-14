"""
This script trains a machine learning model to classify text messages into multiple categories.
It uses a pipeline that includes tokenization, TF-IDF transformation, and a multi-output classifier.
The script loads data from an SQLite database, preprocesses it, performs a grid search to find the best
model parameters, evaluates the model, and saves the trained model to a pickle file.

Usage:
    python train_classifier.py

Functions:
    load_data(database_filepath): Loads data from the specified SQLite database.
    tokenize(text): Tokenizes, case normalizes, and lemmatizes the input text.
    build_model(): Builds a machine learning pipeline and returns a GridSearchCV object.
    evaluate_model(model, X_test, Y_test, category_names): Evaluates the model on the test data and prints the classification report and accuracy for each category.
    save_model(model, model_filepath): Saves the trained model to a pickle file.
    main(): The main function that orchestrates the data loading, model training, evaluation, and saving.

Dependencies:
    - pandas
    - sqlite3
    - nltk
    - scikit-learn
    - pickle
    - re

Make sure to have the following NLTK data downloaded:
    - punkt
    - wordnet

Example:
    To run the script, use the following command:
    python train_classifier.py
"""
from ast import main
import sys
from unicodedata import category
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import sqlite3
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    """
    Load data from a SQLite database, extract features and labels, and return them along with category names.

    Args:
    database_filepath (str): File path to the SQLite database file.

    Returns:
    X (pandas.Series): Messages as features.
    Y (pandas.DataFrame): Target labels.
    categories (Index): Category names for the labels.

    The function performs the following steps:
    1. Connects to the SQLite database and reads the 'DisasterResponse' table into a DataFrame.
    2. Prints a snapshot of the data before and after processing.
    3. Extracts the 'message' column as features (X).
    4. Drops unnecessary columns to create the target labels DataFrame (Y).
    5. Extracts the column names of Y as category names.
    
    Example:
    X, Y, categories = load_data('path/to/database')

    Prints:
    A snapshot of the data before and after dropping specific columns.
    """
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponse', conn)
    
    # Snapshot of the data before processing
    print("Snapshot of the data before processing:")
    print(df.head())
    
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    print("Size before modelling",Y.shape)
    #------------
    print("missing values", Y.isna().sum())
     # Drop rows with NaN values in Y
    X = X[~Y.isnull().any(axis=1)]
    Y = Y.dropna()
    #---------------
    print("Y shape is:", Y.shape)
    print("Any missing values?", Y.isna().sum())
    categories = Y.columns
    
    # Snapshot of the data after processing
    print("Snapshot of the data after processing:")
    print(df.head())
    
    return X, Y, categories

   

def tokenize(text):
    """
        Tokenize, case normalize, and lemmatize the text data.
        
        Parameters:
        text (str): The text data to process.
        
        Returns:
        list: The list of processed tokens.
        """
    # Normalize text to lowercase
    text = text.lower()
    
    # Remove punctuation and non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return lemmatized_tokens

def build_model():
    """
    Build a machine learning pipeline and return a GridSearchCV model.

    Returns:
    GridSearchCV: Cross-validated model with a pipeline.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Define parameters for GridSearchCV
    parameters = {
        'vect__max_df': [0.75, 1.0],
        'clf__estimator__n_estimators': [50, 100]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    print("Pipeline steps initialized.")
    print(f"Pipeline: {pipeline}")
    print("Parameters for GridSearchCV:")
    print(parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, categories):
    """
    Predict and evaluate the model.

    Args:
    model: Trained model to be evaluated.
    X_test (pandas.DataFrame): Test features.
    Y_test (pandas.DataFrame): True labels for test data.
    categories (list): List of category names.

    Returns:
    None
    """
    Y_pred = model.predict(X_test)  # predictions
    
    # Print classification report for each category
    for i in range(len(categories)):
        print(f"Category: {categories[i]}\n", classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    """Save the trained model for future use 

    Args:
        A Machine learning model to be evaluated
        X_test: Test features as defined above.
        Y_test: True labels for the test data.

    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, categories  = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building the model')
        model = build_model()
        
        print('Training the model ')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, categories)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        
        print('The trained model has been saved!')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
        
if __name__ == '__main__':
    main()