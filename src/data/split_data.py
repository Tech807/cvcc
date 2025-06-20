import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

os.makedirs("data/processed",exist_ok=True)

test_size = yaml.safe_load(open("params.yaml"))["split_data"]["test_size"]

df = pd.read_csv("data/raw/drug200.csv")

X = df.drop(columns=["Drug"])
y = df["Drug"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

print(X_test.shape, y_test.shape)


X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
print("Data split and saved successfully.")