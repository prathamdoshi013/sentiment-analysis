# Sentiment Analysis with BERT

This project implements sentiment analysis using BERT (Bidirectional Encoder Representations from Transformers) for classifying text data into positive or negative sentiment.

## Overview

The project consists of two main components:

1. **app.py**: This file contains the Flask web application that allows users to input text and get real-time sentiment predictions using a pre-trained BERT model.

2. **sentiment_analysis.ipynb**: This Jupyter Notebook file contains the code for training the sentiment analysis model using BERT. It includes data loading, preprocessing, model training, evaluation, and saving the trained model.

## Setup

To run the Flask web application, follow these steps:

1. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the Flask application:
   ```
   python app.py
   ```

The web application will be accessible at `http://localhost:5000`.

## Training the Model

To train the sentiment analysis model using BERT, follow the steps outlined in the `sentiment_analysis.ipynb` notebook. Make sure you have the required dataset (`sentiment_train.csv`) available in the project directory.

## Requirements

- Python 
- PyTorch
- Transformers
- Flask
- pandas
- scikit-learn
- Jupyter Notebook (for training the model)

## Acknowledgments

- This project utilizes the BERT model implemented in the Hugging Face `transformers` library.
- The sentiment analysis dataset used for training the model can be found in repository.
