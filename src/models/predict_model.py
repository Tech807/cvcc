import pickle 
import pandas as pd
from sklearn.metrics import accuracy_score
import mlflow
import dagshub

dagshub.init(repo_owner='Tech807', repo_name='cvcc', mlflow=True)

X_test = pd.read_csv("data/interim/X_test_transformed.csv")
y_test = pd.read_csv("data/interim/y_test_transformed.csv")




mlflow.set_tracking_uri("https://dagshub.com/Tech807/cvcc.mlflow")

mlflow.set_experiment("Final_Model")

with mlflow.start_run():
    
    model = pickle.load(open("models/random_forest_model.pkl", "rb"))
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test,y_pred)
    
    mlflow.log_metric("accuracy",accuracy)
    
    mlflow.log_artifact(__file__)
    
    mlflow.sklearn.log_model(model,"model")
    
    if hasattr(model, 'get_params'):
                params = model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)