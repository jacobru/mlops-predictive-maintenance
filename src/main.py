from fastapi import FastAPI
import xgboost as xgb
import pandas as pd
from pydantic import BaseModel

# 1. Define the "Shape" of the data we expect
class MachineData(BaseModel):
    Type: int
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: int
    Torque: float
    Tool_wear: int

# 2. Initialize FastAPI and Load the Model
app = FastAPI(title="Predictive Maintenance API")
model = xgb.Booster()
model.load_model("models/model.json")

@app.get("/")
def home():
    return {"message": "Maintenance API is Running"}

@app.post("/predict")
def predict(data: MachineData):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([data.dict().values()], 
                            columns=['Type', 'Air temperature K', 'Process temperature K', 
                                     'Rotational speed rpm', 'Torque Nm', 'Tool wear min'])
    
    # XGBoost Prediction
    dmatrix = xgb.DMatrix(input_df)
    prediction = model.predict(dmatrix)
    
    # If prediction > 0.5, the machine is likely to fail
    result = "Failure" if prediction[0] > 0.5 else "No Failure"
    
    return {
        "prediction": float(prediction[0]),
        "status": result
    }