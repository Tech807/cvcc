import mlflow
from mlflow.tracking import MlflowClient

# Set your DagsHub MLflow Tracking URI
mlflow.set_tracking_uri("https://dagshub.com/Tech807/cvcc.mlflow")

# Authenticate using your token (optional for private repos)
# export MLFLOW_TRACKING_USERNAME="<your_username>"
# export MLFLOW_TRACKING_PASSWORD="<your_token>"

# Initialize MLflow Client
client = MlflowClient()

# Define model details
model_name = "rf_model"  # case-sensitive
version = "1"  # model version to update
new_stage = "Production"  # Or 'Staging', 'Archived', 'None'

# Move model version to a new stage
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage=new_stage,
    archive_existing_versions=True  # Optional: archive any current model in that stage
)

print(f"Model version {version} of '{model_name}' moved to stage '{new_stage}'")

