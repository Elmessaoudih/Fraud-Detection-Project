import pickle

def train_model(file_path, target_column):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    try:
        # Load and preprocess the dataset
        data = pd.read_csv(file_path)
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train the model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Save the model
        with open("models/trained_model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Save feature names
        with open("models/feature_names.pkl", "wb") as f:
            pickle.dump(X_train.columns.tolist(), f)

        # Evaluate the model
        y_pred = model.predict(X_test)
        metrics = classification_report(y_test, y_pred, output_dict=True)

        return metrics
    except Exception as e:
        raise ValueError(f"Error during training: {e}")
