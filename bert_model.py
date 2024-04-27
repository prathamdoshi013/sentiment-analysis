import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report
import pickle
# Load data
df = pd.read_csv('sentiment_train.csv')

# Train-validation-test split
train_text, temp_text, train_labels, temp_labels = train_test_split(df['sentence'], df['label'],
                                                                    random_state=2021,
                                                                    test_size=0.3,
                                                                    stratify=df['label'])

val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=2021,
                                                                test_size=0.5,
                                                                stratify=temp_labels)

# TF-IDF vectorization
max_features = 10000  # Adjust max_features as needed
tfidf_vectorizer = TfidfVectorizer(max_features=max_features)  
train_tfidf = tfidf_vectorizer.fit_transform(train_text)
val_tfidf = tfidf_vectorizer.transform(val_text)
test_tfidf = tfidf_vectorizer.transform(test_text)
# Save the TfidfVectorizer
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)

# Get the actual number of features after vectorization
num_features = min(max_features, train_tfidf.shape[1])  # Ensure consistency with max_features

# Define model architecture
model = Sequential([
    Dense(512, activation='relu', input_shape=(num_features,)),  # Use num_features instead of train_tfidf.shape[1]
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(train_tfidf, train_labels, epochs=10, batch_size=32, validation_data=(val_tfidf, val_labels))

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_tfidf, test_labels)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Generate predictions
test_preds = (model.predict(test_tfidf) > 0.5).astype("int32")

# Print classification report
print(classification_report(test_labels, test_preds))

# Save the model
model.save("tfidf_sentiment_model.h5")
