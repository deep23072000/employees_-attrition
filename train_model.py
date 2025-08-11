import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# =======================
# Load Dataset
# =======================
DATA_FILE = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(DATA_FILE)

y = df["Attrition"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Attrition", "EmployeeNumber"], errors="ignore")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Preprocessing
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
])
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "attrition_pipeline.joblib")

# Save schema
schema = {
    "fields": [
        {
            "name": col,
            "dtype": "numeric" if col in num_cols else "categorical",
            "sample_values": X[col].dropna().unique().tolist()[:10]
        }
        for col in X.columns
    ]
}
with open("schema.json", "w") as f:
    json.dump(schema, f, indent=2)

print("✅ Model and schema saved successfully.")
