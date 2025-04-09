from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

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

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            input_data = [float(request.form[col]) for col in columns]
            input_df = pd.DataFrame([input_data], columns=columns)
            input_scaled = scaler.transform(input_df)
            prob = model.predict_proba(input_scaled)[0][1]
            result = 1 if prob >= threshold else 0
            prediction = str(result)
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template("index.html", columns=columns, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
