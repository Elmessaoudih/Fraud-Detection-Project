import pandas as pd
import os

def preprocess_data(file_path):
    try:
        # Load the data
        data = pd.read_csv(file_path)

        # Handle missing values
        # Fill missing numeric values with their mean
        for col in data.select_dtypes(include=["float64", "int64"]).columns:
            data[col].fillna(data[col].mean(), inplace=True)

        # Fill missing categorical values with "Unknown" and encode them
        for col in data.select_dtypes(include=["object"]).columns:
            data[col].fillna("Unknown", inplace=True)
            data[col] = pd.factorize(data[col])[0]

        # Save the cleaned data
        cleaned_dir = "data/cleaned_data"
        os.makedirs(cleaned_dir, exist_ok=True)
        cleaned_file_path = os.path.join(cleaned_dir, "FraudDetection_Preprocessing.csv")
        data.to_csv(cleaned_file_path, index=False) # Save the cleaned file
        return cleaned_file_path
    except Exception as e:
        raise ValueError(f"Error processing file: {e}") # Throw an error if something goes wrong
