import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
import sqlite3
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens

# Load data
try:
    database_filepath = "data/DisasterResponse.db"
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponse', conn)
    conn.close()
    logger.info("Data loaded successfully.")
    logger.info(f"Data shape: {df.shape}")
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise

# Load model
try:
    model = joblib.load("models/classifier.pkl")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

@app.route('/')
@app.route('/index')
def index():
    """
    Extract data for visualizations and render the homepage with interactive plots.

    Returns:
        Rendered HTML page with Plotly visualizations.
    """
    try:
        # Extract data needed for visuals
        logger.info("Extracting data for visuals.")
        genre_counts = df['genre'].value_counts()
        genre_names = genre_counts.index.tolist()

        response_counts = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum().sort_values(ascending=False).head(10)
        response_names = response_counts.index.tolist()

        # Create visuals
        logger.info("Creating visuals.")
        graphs = [
            {
                'data': [
                    Bar(
                        x=genre_names,
                        y=genre_counts
                    )
                ],
                'layout': {
                    'title': 'Distribution of Message Genres',
                    'yaxis': {'title': "Count"},
                    'xaxis': {'title': "Genre"}
                }
            },
            {
                'data': [
                    Bar(
                        x=response_names,
                        y=response_counts
                    )
                ],
                'layout': {
                    'title': 'Top 10 Message Categories',
                    'yaxis': {'title': "Count"},
                    'xaxis': {'title': "Category"}
                }
            },
            {
                'data': [
                    Pie(
                        labels=genre_names,
                        values=genre_counts,
                        hole=.3
                    )
                ],
                'layout': {
                    'title': 'Proportion of Message Genres'
                }
            }
        ]

        # Encode plotly graphs in JSON
        logger.info("Encoding visuals to JSON.")
        ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
        graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

        # Render web page with plotly graphs
        logger.info("Rendering web page with visuals.")
        return render_template('master.html', ids=ids, graphJSON=graphJSON)

    except Exception as e:
        logger.error(f"Error in index route: {e}")
        return "An error occurred while rendering the page."

@app.route('/go')
def go():
    """
    Render the classification result page for user queries.

    Returns:
        Rendered HTML page with classification results.
    """
    try:
        query = request.args.get('query', '')
        classification_labels = model.predict([query])[0]
        classification_results = dict(zip(df.columns[4:], classification_labels))

        return render_template(
            'go.html',
            query=query,
            classification_result=classification_results
        )
    except Exception as e:
        logger.error(f"Error in go route: {e}")
        return "An error occurred while processing your request."

def main():
    app.run(host='0.0.0.0', port=3002, debug=False)  # Turn off debug for production

if __name__ == '__main__':
    main()
