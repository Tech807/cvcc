import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import os

os.makedirs("data/interim",exist_ok=True)

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")


transformer = ColumnTransformer(
    transformers=[
        ("scaler",StandardScaler(),["Age","Na_to_K"]),
        ("one-hot-encoding",OneHotEncoder(drop="first",sparse_output=False),["Cholesterol"]),
        ("ordinal_encoder",OrdinalEncoder(categories=[["F","M"],['LOW', 'NORMAL', 'HIGH']]),['Sex','BP'])
    ],remainder="passthrough"
)

X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)

X_train_new = pd.DataFrame(X_train_transformed,columns=transformer.get_feature_names_out())
X_test_new = pd.DataFrame(X_test_transformed,columns=transformer.get_feature_names_out())

scaler = LabelEncoder()

# Flatten y to 1D array
y_train_new = scaler.fit_transform(y_train.values.ravel())
y_test_new = scaler.transform(y_test.values.ravel())


X_train_new.to_csv("data/interim/X_train_transformed.csv", index=False)
X_test_new.to_csv("data/interim/X_test_transformed.csv", index=False)
pd.DataFrame(y_train_new, columns=["target"]).to_csv("data/interim/y_train_transformed.csv", index=False)
pd.DataFrame(y_test_new, columns=["target"]).to_csv("data/interim/y_test_transformed.csv", index=False)

print("Data transformation complete. Files saved in 'data/interim'")