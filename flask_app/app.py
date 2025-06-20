from flask import Flask, render_template, request
import mlflow.pyfunc
import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Flask app setup
app = Flask(__name__)

# Load MLflow model from DagsHub
mlflow.set_tracking_uri("https://dagshub.com/Tech807/cvcc.mlflow")
model = mlflow.pyfunc.load_model("models:/rf_model@production")

# Preprocessing pipeline (same as training)
transformer = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), ["Age", "Na_to_K"]),
        ("one-hot-encoding", OneHotEncoder(drop="first", sparse_output=False), ["Cholesterol"]),
        ("ordinal_encoder", OrdinalEncoder(categories=[["F", "M"], ["LOW", "NORMAL", "HIGH"]]), ["Sex", "BP"]),
    ],
    remainder="passthrough"
)

# Fit transformer using X_train (needed for consistent behavior)
X_train = pd.read_csv("data/processed/X_train.csv")
transformer.fit(X_train)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get and map form values
    age = float(request.form["Age"])
    sex = "M" if request.form["Sex"] == "Male" else "F"
    bp = request.form["BP"].upper()
    cholesterol = request.form["Cholesterol"].upper()
    na_to_k = float(request.form["Na_to_K"])

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "BP": bp,
        "Cholesterol": cholesterol,
        "Na_to_K": na_to_k
    }])

    # Apply transformation
    transformed = transformer.transform(input_data)

    # Predict
    prediction = model.predict(transformed)

    # Map label index to name
    labels = ["DrugY", "drugA", "drugB", "drugC", "drugX"]
    predicted_label = labels[int(prediction[0])] if isinstance(prediction[0], (int, np.integer)) else prediction[0]

    return render_template("index.html", prediction=predicted_label)


if __name__ == "__main__":
    app.run(debug=True)
