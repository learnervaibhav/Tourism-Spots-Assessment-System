from flask import Flask, request,render_template, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model and tokenizer
model_name = "D:\\Sentiment analysis model"  # Ensure this is the correct path to your model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set up the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Path to the large dataset file (CSV or other format)
dataset_path = "D:\Sentiment analysis model\Review_db.csv"  # Ensure this is the correct path to your dataset

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json  # Get the JSON data from the request
    city = data.get("City")  # Extract city
    place = data.get("Place")  # Extract place

    # Check if city and place are provided
    if not city or not place:
        return jsonify({"error": "Please provide both city and place"}), 400

    # Load only relevant rows from the dataset
    try:
        # Use the relevant columns and avoid memory issues
        reviews_df = pd.read_csv(
            dataset_path,
            usecols=['City', 'Place', 'Review'],  # Ensure these columns exist in your dataset
            low_memory=False
        )

        # Filter for the specified city and place
        filtered_reviews = reviews_df[(reviews_df['City'] == city) & (reviews_df['Place'] == place)]

        if filtered_reviews.empty:
            return jsonify({"error": "No reviews found for the specified city and place"}), 404

        # Get reviews as a list
        reviews = filtered_reviews['Review'].tolist()

    except Exception as e:
        return jsonify({"error": f"Failed to load data: {str(e)}"}), 500

    # Analyze each review's sentiment and aggregate scores
    scores = []
    for review in reviews:
        result = sentiment_analyzer(review)[0]
        # Convert score to positive or negative value
        score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
        scores.append(score)

    # Calculate average sentiment score and determine overall sentiment
    if scores:  # Ensure there are scores to calculate the average
        avg_score = np.mean(scores)
        overall_sentiment = "POSITIVE" if avg_score > 0 else "NEGATIVE"
    else:
        avg_score = 0  # If no scores, set to 0
        overall_sentiment = "NEUTRAL"  # Neutral if no reviews were analyzed

    # Return the aggregated sentiment result for the given city and place
    return jsonify({
        "City": city,
        "Place": place,
        "average_sentiment_score": avg_score,
        "overall_sentiment": overall_sentiment,
        "review_count": len(reviews)
    })

if __name__ == '__main__':
    app.run(debug=True)
