from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model, scaler, threshold
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("threshold.txt", "r") as f:
    threshold = float(f.read())

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", columns=columns)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        # Ensure the data contains the attribute counts
        input_count = data.get("input", 0)
        output_count = data.get("output", 0)
        for_count = data.get("for", 0)
        if_count = data.get("if", 0)
        while_count = data.get("while", 0)
        exp_count = data.get("Exp", 0)
        fc_count = data.get("fc", 0)

        # Example prediction logic (this should be replaced with your actual logic or model)
        # Here, just a simple rule for demo purposes:
        if input_count > 0 and output_count > 0:
            prediction = 1  # Low risk
        else:
            prediction = 0  # High risk

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
