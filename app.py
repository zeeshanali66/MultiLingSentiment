from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Initialize the Flask app
app = Flask(__name__)

# --- LOAD MODEL & TOKENIZER ---
MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# --- HELPER FUNCTION ---
def analyze_sentiment(user_input):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_map = {
        0: "Very Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Very Positive"
    }
    prediction = torch.argmax(probabilities, dim=-1).item()
    return sentiment_map.get(prediction, "Unknown")

# --- API ROUTES ---
@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "model": MODEL_NAME,
        "message": "API is ready to receive requests at /detect_emotion"
    })

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion_route():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_input = data.get('text')

    if not user_input:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    try:
        emotion = analyze_sentiment(user_input)
        return jsonify({"emotion": emotion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
