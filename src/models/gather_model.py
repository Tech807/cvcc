import mlflow.pyfunc
import mlflow
import pandas as pd

data = pd.read_csv("data/interim/X_test_transformed.csv")

mlflow.set_tracking_uri("https://dagshub.com/Tech807/cvcc.mlflow")

model = mlflow.pyfunc.load_model("models:/rf_model@production")

result = model.predict(data)

print(result)