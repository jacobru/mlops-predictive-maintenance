import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import json

def train():
    # 1. Load the processed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")

    # 2. Start an MLflow Experiment
    # This creates a record of this specific run
    mlflow.set_experiment("Predictive_Maintenance")

    with mlflow.start_run():
        # Define model parameters
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42
        }
        
        # 3. Initialize and Train XGBoost
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # 4. Predict and Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {acc:.4f}")

        # 5. Log everything to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.xgboost.log_model(model, "model")

        # Save model locally for our API later
        os.makedirs("models", exist_ok=True)
        model.save_model("models/model.json")
        
        # Save metrics for DVC to track
        with open("reports/metrics.json", "w") as f:
            json.dump({"accuracy": acc}, f)

if __name__ == "__main__":
    # Create reports folder if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    train()