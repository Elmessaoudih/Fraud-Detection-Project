from flask import Flask, render_template, request, jsonify
from utils.preprocessing import preprocess_data
from utils.training import train_model
from utils.prediction import make_prediction
import os
import pandas as pd

app = Flask(__name__)

# Ensure directories exist
os.makedirs("data/cleaned_data", exist_ok=True)
os.makedirs("models", exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded."}), 400
        file_path = os.path.join("data", file.filename)
        file.save(file_path)
        cleaned_path = preprocess_data(file_path)
        return jsonify({"message": "File uploaded and preprocessed successfully.", "cleaned_path": cleaned_path})
    return render_template('upload.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        file = request.files.get('file')
        target_column = request.form.get('target_column')
        if not file or not target_column:
            return jsonify({"error": "File or target column missing."}), 400
        file_path = os.path.join("data/cleaned_data", file.filename)
        file.save(file_path)
        metrics = train_model(file_path, target_column)
        return jsonify({"message": "Training completed successfully!", "metrics": metrics})
    return render_template('train.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded."}), 400
        file_path = os.path.join("data", file.filename)
        file.save(file_path)
        predictions = make_prediction(file_path)
        return jsonify({"predictions": predictions})
    return render_template('predict.html')

@app.route('/get_columns', methods=['POST'])
def get_columns():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded."}), 400

        # Save the uploaded file temporarily
        file_path = os.path.join("data/temp", file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)

        # Read the dataset and get column names
        data = pd.read_csv(file_path)
        columns = data.columns.tolist()

        # Remove the temporary file after reading
        os.remove(file_path)

        return jsonify({"columns": columns})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
