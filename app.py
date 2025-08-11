from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import json

app = Flask(__name__)

# Load model + schema
model = joblib.load("attrition_pipeline.joblib")
with open("schema.json") as f:
    schema = json.load(f)

@app.route("/")
def index():
    return render_template("index.html", schema=schema)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data], columns=[f["name"] for f in schema["fields"]])
    pred = model.predict(df)[0]
    prob = float(model.predict_proba(df)[:, 1][0])
    return jsonify({
        "attrition": "Yes" if pred == 1 else "No",
        "probability": prob
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
