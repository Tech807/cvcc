import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

def load_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        print("Data loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def main():
    url = "https://raw.githubusercontent.com/MainakRepositor/Datasets/refs/heads/master/drug200.csv"
    try:
        df = load_data(url)
        df.to_csv("data/raw/drug200.csv", index=False)
        print("Data saved to data/raw/drug200.csv")
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()