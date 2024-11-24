from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd
import torch

app = Flask(__name__)

# Load the RoBERTa model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Load the reviews data
review_df = pd.read_csv("..\review_db_limited.csv")
density_df = pd.read_csv("..\Density.csv")

# Convert City and Name columns to lowercase for case-insensitive search
density_df['City'] = density_df['City'].str.lower()
density_df['Name'] = density_df['Name'].str.lower()


# Function to calculate sentiment scores
def get_sentiment_scores(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = outputs.logits.detach().numpy()[0]
    probs = softmax(scores)
    return {"Positive": probs[2], "Neutral": probs[1], "Negative": probs[0]}


# Aggregate sentiments by city and place
def aggregate_sentiments(city, place):
    filtered_df = review_df[
        (review_df['City'].str.lower() == city.lower()) & (review_df['Place'].str.lower() == place.lower())]
    if filtered_df.empty:
        return None
    filtered_df['Sentiment'] = filtered_df['Review'].apply(get_sentiment_scores)
    sentiment_scores = pd.json_normalize(filtered_df["Sentiment"]).mean().to_dict()
    return sentiment_scores


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/get_sentiment', methods=['POST'])
def get_sentiment():
    data = request.json
    city = data.get("City")
    place = data.get("Place")

    sentiment_scores = aggregate_sentiments(city, place)
    if sentiment_scores:
        result = {
            "City": city,
            "Place": place,
            "Sentiment": sentiment_scores
        }
        return jsonify(result)
    else:
        return jsonify({"error": "No reviews found for this city and place"}), 404


@app.route('/get_features', methods=['POST'])
def get_features():
    data = request.form
    city = data.get("city").lower()
    place = data.get("name").lower()

    # Search in density_df
    features = density_df[(density_df['City'] == city) & (density_df['Name'] == place)].to_dict(orient="records")
    if features:
        return jsonify(features[0])
    else:
        return jsonify({"error": "No tourist place information found for this city and place"}), 404


if __name__ == '__main__':
    app.run(debug=True)
