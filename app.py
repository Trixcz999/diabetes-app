import os
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

# Load the saved model and scaler
model = joblib.load(os.path.join(os.path.dirname(__file__), 'diabetes_model.pkl'))
scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'scaler.pkl'))

# Show the home page
@app.route('/')
def home():
    return render_template('index.html')

# When the form is submitted
@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['pregnancies']),
        float(request.form['glucose']),
        float(request.form['bloodpressure']),
        float(request.form['skinthickness']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['dpf']),
        float(request.form['age'])
    ]

    # Scale the input
    features_scaled = scaler.transform([features])

    # Make prediction
    prediction = model.predict(features_scaled)[0]

    if prediction == 1:
        result = "Diabetic"
    else:
        result = "Non-Diabetic"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)