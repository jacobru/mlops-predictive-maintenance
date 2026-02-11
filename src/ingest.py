import pandas as pd
import os

def download_data():
    # The URL to the raw CSV data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    
    print("Downloading dataset...")
    df = pd.read_csv(url)
    
    # Create the path if it doesn't exist (extra safety)
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    
    # Save the data
    raw_data_path = os.path.join("data", "raw", "predictive_maintenance.csv")
    df.to_csv(raw_data_path, index=False)
    print(f"Data saved successfully to {raw_data_path}")

if __name__ == "__main__":
    download_data()