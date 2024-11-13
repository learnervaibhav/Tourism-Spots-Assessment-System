from flask import Flask, request, jsonify, render_template
import pandas as pd

app = Flask(__name__)

# Load the combined sentiment scores DataFrame
sentiment_df = pd.read_csv("D:\\Sentiment analysis model\\vaderroberta.csv")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/get_sentiment', methods=['POST','GET'])
def get_sentiment():
    data = request.get_json()
    city = data.get("City")
    place = data.get("Place")
    model_choice = data.get("Model")

    # Validating input fields
    if not city or not place or not model_choice:
        return jsonify({"error": "Please provide city, place, and model choice"}), 400

    # Filtering DataFrame for specified city and place
    try:
        filtered = sentiment_df[(sentiment_df['City'] == city) & (sentiment_df['Place'] == place)]
        if filtered.empty:
            return jsonify({"error": "No sentiment score found for the specified city and place"}), 404


        if model_choice == "VADER":
            result = {
                "City": city,
                "Place": place,
                "Sentiment": {
                    "Positive": filtered['vaderpos'].values[0],
                    "Neutral": filtered['vaderneu'].values[0],
                    "Negative": filtered['vaderneg'].values[0],
                    "Compound": filtered['vadercompound'].values[0]
                }
            }
        elif model_choice == "RoBERTa":
            result = {
                "City": city,
                "Place": place,
                "Sentiment": {
                    "Positive": filtered['roberta_pos'].values[0],
                    "Neutral": filtered['roberta_neu'].values[0],
                    "Negative": filtered['roberta_neg'].values[0],
                }
            }
        else:
            return jsonify({"error": "Invalid model choice"}), 400

    except Exception as e:
        return jsonify({"error": f"Failed to retrieve data: {str(e)}"}), 500

    # Return sentiment data for the given city, place, and model
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
