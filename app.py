from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModel, BertTokenizerFast
from bert_model import BERTClassifier

app = Flask(__name__, template_folder='templates')

# Load pre-trained BERT model
bert = AutoModel.from_pretrained('bert-base-uncased')

# Create an instance of BERTClassifier with the pre-trained BERT model
model = BERTClassifier(bert)

# Load the saved model weights
model.load_state_dict(torch.load(r'project\sentiment analysis\bert_sentiment_model.pth'))
model.eval()

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Define route for root URL
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'text' in request.form:
            text = request.form['text']
            prediction = predict_sentiment(text)
            return render_template('index.html', prediction=prediction, text=text)
    return render_template('index.html', prediction=None, text=None)

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    prediction = predict_sentiment(text)
    return jsonify({'prediction': prediction})

def predict_sentiment(text):
    # Tokenize and encode the input text
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    
    # Perform inference
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        _, preds = torch.max(outputs, dim=1)
    
    # Convert predictions to labels
    labels = ['Negative', 'Positive']
    pred_label = labels[preds.item()]
    
    return pred_label

if __name__ == '__main__':
    app.run(debug=True)
