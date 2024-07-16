import os
from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the form
        bilirubin = float(request.form['bilirubin'])
        alk_phosphate = float(request.form['alk_phosphate'])
        sgot = float(request.form['sgot'])
        varices = int(request.form['varices'])
        albumin = float(request.form['albumin'])
        
        # Create a feature array in the correct order expected by the model
        features = np.array([[bilirubin, alk_phosphate, sgot, varices, albumin]])
        
        # Make a prediction
        prediction = model.predict(features)
        
        # Return the result to the template
        return render_template('index.html', result=prediction[0])
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
