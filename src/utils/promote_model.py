"""
Model Promotion Script
Promotes the latest registered model to 'Production' stage in MLflow.
"""

import os
import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger

# Use env var so CI/CD works without code changes
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

def promote_latest_model(model_name: str):
    """Transition the latest 'None' stage model to 'Production'."""
    client = MlflowClient()
    
    # Get latest version
    try:
        versions = client.get_latest_versions(model_name, stages=["None"])
        if not versions:
            logger.warning(f"No versions found for model {model_name} in 'None' stage.")
            return

        latest_version = versions[0].version
        logger.info(f"Promoting model {model_name} version {latest_version} to Production...")
        
        # Transition
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.success(f"Model {model_name} v{latest_version} is now in Production.")
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")

if __name__ == "__main__":
    import yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    promote_latest_model(params["train"]["model_name"])
