from flask import Flask, request, jsonify
from gradio_client import Client
import os

app = Flask(__name__)

# Load your Gradio Space as a client
client = Client("zeeshanali66/sentiment-analysis")

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "message": "Send POST to /predict with JSON {\"text\": \"your message\"}"
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_input = data.get("text")

    if not user_input:
        return jsonify({"error": "Missing 'text' field in JSON"}), 400

    try:
        # Call the Gradio Space
        result = client.predict(user_input=user_input, api_name="/predict")
        return jsonify({"sentiment": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
