# Sentiment Analysis with TF-IDF and Deep Learning

This project demonstrates sentiment analysis using TF-IDF vectorization and deep learning techniques. It involves training a deep learning model on a dataset of text data labeled with sentiment, then deploying the model using Streamlit for real-time sentiment analysis.

## Files Included:

1. **`train_model.py`**: This script is responsible for training the deep learning model using TF-IDF vectorized text data.

2. **`app.py`**: This script contains the Streamlit application for performing real-time sentiment analysis using the trained model.

3. **`sentiment_train.csv`**: CSV file containing the dataset for training the sentiment analysis model. It consists of two columns: `sentence` (containing the text data) and `label` (containing the sentiment label).

4. **`tfidf_sentiment_model.h5`**: The trained deep learning model saved in HDF5 format.

5. **`tfidf_vectorizer.pkl`**: The TF-IDF vectorizer trained on the text data, saved using pickle.

## Setup Instructions:

1. **Install Dependencies**: Ensure you have the required dependencies installed. You can install them using pip:

   ```
   pip install pandas scikit-learn tensorflow streamlit
   ```

2. **Training the Model**: Run the `train_model.py` script to train the sentiment analysis model. This script will preprocess the text data, train the deep learning model, and save both the model and TF-IDF vectorizer.

   ```
   python train_model.py
   ```

3. **Running the Streamlit App**: Run the `app.py` script to launch the Streamlit web application for real-time sentiment analysis.

   ```
   streamlit run app.py
   ```

4. **Interacting with the Application**: Once the Streamlit app is running, enter a sentence into the text input field and click the "Analyze Sentiment" button. The application will use the pre-trained model to predict the sentiment of the entered text and display the result.

## Additional Notes:

- The `train_model.py` script can be modified to adjust model hyperparameters, such as the number of epochs, batch size, and neural network architecture.

- The performance of the sentiment analysis model can be evaluated using metrics such as accuracy, precision, recall, and F1-score. These metrics are computed using the `classification_report` function from scikit-learn.

- Feel free to customize the Streamlit application (`app.py`) to include additional features or visualizations as needed.
