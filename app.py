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
        alkaline_phosphate = float(request.form['alk_phosphate'])
        sgot = float(request.form['sgot'])
        varices = float(request.form['varices'])
        albumin = float(request.form['albumin'])

        # Prepare the features for prediction
        final_features = [
            np.array([bilirubin, alkaline_phosphate, sgot, varices, albumin])]

        # Make prediction
        prediction = model.predict(final_features)

        # Output the prediction
        output = prediction[0]
        return render_template('index.html', result=f'Predicted Value: {output}')
    except Exception as e:
        return str(e)


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
