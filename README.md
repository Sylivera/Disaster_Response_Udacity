
---

# Disaster Response Pipeline Project

This project is part of the Udacity Data Scientist Nanodegree program. The goal of this project is to build a machine learning pipeline to categorize real messages that were sent during disaster events. The project includes a web app where an emergency worker can input a new message and get classification results in multiple categories. The web app also displays visualizations of the data.
Please use this link to access the repo - https://github.com/Sylivera/Disaster_Response_Udacity.git

## Table of Contents

1. [Project Summary](#project-summary)
2. [Files in the Repository](#files-in-the-repository)
3. [Installation](#installation)
4. [Running the Python Scripts](#running-the-python-scripts)
5. [Running the Web App](#running-the-web-app)
6. [Function Documentation](#function-documentation)

## Project Summary

The Disaster Response Pipeline project includes:
- ETL Pipeline: A data pipeline that extracts data from a source, cleans it, and stores it in a database.
- ML Pipeline: A machine learning pipeline that loads data from the database, trains a model, and saves the trained model to a file.
- Web App: A Flask web application that serves a web interface to classify disaster response messages and visualize data.

## Files in the Repository

- `app`
  - `run.py`: Flask file to run the web application.
  - `templates`
    - `go.html`: HTML template for the classification result page.
    - `master.html`: HTML template for the main page.

- `data`
  - `categories.csv`: Data file containing the categories of messages.
  - `messages.csv`: Data file containing the messages.
  - `DisasterResponse.db`: SQLite database containing the cleaned data.
  - `process_data.py`: Python script to process the data and save it to the database.

- `models`
  - `classifier.pkl`: Trained model saved as a pickle file.
  - `train_classifier.py`: Python script to train the model and save it.

- `README.md`: This README file.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/disaster-response-pipeline
    cd disaster-response-pipeline
    ```

2. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

3. Ensure you have the necessary NLTK data:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    ```

## Running the Python Scripts

### 1. Process Data

The `process_data.py` script performs the ETL process. It loads messages and categories datasets, cleans the data, and stores it in an SQLite database.

To run the script, use:
```sh
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

### 2. Train Classifier

The `train_classifier.py` script trains the machine learning model and saves it as a pickle file.

To run the script, use:
```sh
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

## Running the Web App

To run the web app, navigate to the `app` directory and run the Flask server:
```sh
cd app
python run.py
```

Open your browser and go to `http://0.0.0.0:3002/` to use the web app.

## Function Documentation

### `process_data.py`

- `load_data(messages_filepath, categories_filepath)`: Loads messages and categories datasets and merges them.
- `clean_data(df)`: Cleans the merged dataset by splitting categories and removing duplicates.
- `save_data(df, database_filename)`: Saves the cleaned data to an SQLite database.
- `main()`: Main function to run the ETL pipeline.

### `train_classifier.py`

- `load_data(database_filepath)`: Loads data from the SQLite database.
- `custom_tokenize(text)`: Tokenizes, case normalizes, and lemmatizes the input text.
- `build_model()`: Builds a machine learning pipeline and returns a GridSearchCV object.
- `evaluate_model(model, X_test, Y_test, category_names)`: Evaluates the model on the test data and prints the classification report and accuracy for each category.
- `save_model(model, model_filepath)`: Saves the trained model to a pickle file.
- `main()`: Main function to run the machine learning pipeline.

### `run.py`

- `tokenize(text)`: Tokenizes, case normalizes, and lemmatizes the input text.
- `index()`: Renders the main page with Plotly visualizations.
- `go()`: Renders the classification result page for the user query.

---

This README file provides a comprehensive overview of the project, including instructions on how to run the scripts and web app, and an explanation of the files and functions.
