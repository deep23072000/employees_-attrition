import pandas as pd
import json
import joblib
from flask import Flask, request, jsonify, render_template

# =======================
# Load model and schema
# =======================
MODEL_PATH = "attrition_pipeline.joblib"
SCHEMA_PATH = "schema.json"

model = joblib.load(MODEL_PATH)
with open(SCHEMA_PATH) as f:
    schema = json.load(f)

# =======================
# Flask app
# =======================
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", schema=schema)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df_in = pd.DataFrame([data], columns=[f["name"] for f in schema["fields"]])
    pred = model.predict(df_in)[0]
    prob = float(model.predict_proba(df_in)[:,1][0])
    return jsonify({
        "attrition": "Yes" if pred == 1 else "No",
        "probability": prob
    })

if __name__ == "__main__":
    app.run(port=7860, debug=True)
