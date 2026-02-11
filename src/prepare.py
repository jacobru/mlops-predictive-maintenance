import pandas as pd
from sklearn.model_selection import train_test_split
import os

def prepare_data():
    raw_data_path = os.path.join("data", "raw", "predictive_maintenance.csv")
    if not os.path.exists(raw_data_path):
        print("Error: Raw data file not found.")
        return
        
    df = pd.read_csv(raw_data_path)
    
    # NEW: Clean column names to satisfy XGBoost requirements
    df.columns = [col.replace('[', '').replace(']', '').replace('<', '').strip() for col in df.columns]
    
    # Cleaning
    df = df.drop(columns=['UDI', 'Product ID'])
    df['Type'] = df['Type'].map({'L': 0, 'M': 1, 'H': 2})
    
    # Splitting
    X = df.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
    y = df['Machine failure']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    X_train.to_csv(os.path.join("data", "processed", "X_train.csv"), index=False)
    X_test.to_csv(os.path.join("data", "processed", "X_test.csv"), index=False)
    y_train.to_csv(os.path.join("data", "processed", "y_train.csv"), index=False)
    y_test.to_csv(os.path.join("data", "processed", "y_test.csv"), index=False)
    
    print("Data preparation complete. Special characters removed from column names.")

if __name__ == "__main__":
    prepare_data()