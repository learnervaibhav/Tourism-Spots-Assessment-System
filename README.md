# Tourism Spots Assessment System

![Screenshot 2024-11-18 203630](https://github.com/user-attachments/assets/2c5d5201-8067-44ee-bcb1-dc5f37957ee4)

## Overview
The project uses a fine-tuned Roberta model to process sentiment data from reviews. By aggregating sentiment for each city/place, users can see an overview of how a destination is perceived by others, along with specific attributes like visitor ratings, entrance fees, and recommended visit times. 

The data is processed using Flask to create an interactive web application, where users can input a city and place to retrieve sentiment insights and destination details.
 

## Features
- <ins>**Sentiment Analysis:**</ins> Sentiment scores for tourist destinations are based on the processing of the Roberta model.

- <ins>**City and Place Data Retrieval:**</ins> Query tourist destinations by city and place names for relevant details like entrance fees, reviews, ratings, etc.

- <ins>**Web Interface:**</ins> A user-friendly web interface that interacts with the sentiment analysis results.

<br>

## Technology Stack
1. **Python:** Core language used for data processing and analysis.

2.  **Flask:** Micro web framework to manage HTTP requests and render the web interface.

3.  **Roberta Model:** Transformer model for sentiment analysis.

4.  **Pandas:** Data manipulation and aggregation.
HTML & CSS: Web interface components.

<br> 

## Project Structure

```
tourist-sentiment-analysis/
├── app.py                # Main Flask application file
├── requirements.txt      # List of Python packages
├── templates/
│   └── index.html        # HTML template for the web interface
├── static/
│   └── styles.css        # CSS file for web styling
├── Review_db.csv         # Review dataset for sentiment analysis
└── Density.csv           # Dataset with tourist destination details
```
