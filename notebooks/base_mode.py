import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import dagshub
import mlflow

dagshub.init(repo_owner='Tech807', repo_name='cvcc', mlflow=True)

df = pd.read_csv("https://raw.githubusercontent.com/MainakRepositor/Datasets/refs/heads/master/drug200.csv")

X = df.drop(columns=["Drug"])
y = df["Drug"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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



mlflow.set_tracking_uri("https://dagshub.com/Tech807/cvcc.mlflow")

mlflow.set_experiment("Base model")

with mlflow.start_run(run_name="RandomForestClassifier"):
    
    n_estimator = 50
    
    model = RandomForestClassifier(n_estimators=n_estimator)
    
    
    model.fit(X_train_new,y_train_new)
    
    y_pred = model.predict(X_test_new)
    
    accuracy = accuracy_score(y_test_new,y_pred)
    
    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_param("n_estimator",n_estimator)
    
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(model,"model")