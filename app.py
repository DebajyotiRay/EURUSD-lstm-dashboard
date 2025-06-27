from flask import Flask, jsonify, render_template
import subprocess
import os
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    try:
        subprocess.run(['python', 'LSTM_model_prediction.py'], check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({'error': 'Prediction script failed', 'details': str(e)}), 500

    if not os.path.exists("prediction_results.json"):
        return jsonify({'error': 'Prediction output not found'}), 500

    with open("prediction_results.json", "r") as f:
        try:
            results = json.load(f)
        except Exception as e:
            return jsonify({'error': 'Failed to read prediction data', 'details': str(e)}), 500

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)