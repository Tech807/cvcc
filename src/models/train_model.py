import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

os.makedirs("models", exist_ok=True)

X_train = pd.read_csv("data/interim/X_train_transformed.csv")
X_test = pd.read_csv("data/interim/X_test_transformed.csv")
y_train = pd.read_csv("data/interim/y_train_transformed.csv")
y_test = pd.read_csv("data/interim/y_test_transformed.csv")

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train,y_train)


pickle.dump(model, open("models/random_forest_model.pkl", "wb"))
print("Model trained and saved successfully as 'random_forest_model.pkl' in the 'models' directory.")
