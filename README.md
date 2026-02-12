# End-to-End Predictive Maintenance MLOps Pipeline

This project demonstrates a professional-grade MLOps workflow for predicting machine failures. It moves beyond a simple notebook, implementing automated data ingestion, versioning, experiment tracking, and containerized deployment.



## Project Overview
* **Problem:** Predict machine failure based on sensor data (Temperature, Torque, Tool Wear).
* **Goal:** Create a production-ready API that serves a trained XGBoost model.

## Technical Stack
* **Data Versioning:** [DVC](https://dvc.org/) (Data Version Control)
* **Experiment Tracking:** [MLflow](https://mlflow.org/)
* **Model:** XGBoost Classifier
* **API:** FastAPI
* **Containerization:** Docker
* **Language:** Python 3.10+

## 📂 Project Structure
```text
├── data/               # Versioned by DVC (not tracked by Git)
├── models/             # Saved model artifacts (.json)
├── reports/            # Performance metrics
├── src/
│   ├── ingest.py       # Data acquisition
│   ├── prepare.py      # Data cleaning & feature engineering
│   ├── train.py        # Model training & MLflow logging
│   └── main.py         # FastAPI application
├── Dockerfile          # Containerization instructions
└── requirements.txt    # Project dependencies

1. Setup Environment

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

2. Run the Pipeline

python src/ingest.py
python src/prepare.py
python src/train.py

3. Launch the API

uvicorn src.main:app --reload
Visit http://127.0.0.1:8000/docs to interact with the API.

4. Docker Deployment
To run the entire application in a container:

docker build -t maintenance-api .
docker run -p 8000:8000 maintenance-api


