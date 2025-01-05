from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
# Check if breakdown_model is a tuple and unpack if necessary
breakdown_model = joblib.load('model.pkl')  # For breakdown prediction

# If breakdown_model is a tuple, unpack it
if isinstance(breakdown_model, tuple):
    breakdown_model, breakdown_features = breakdown_model

@app.route('/')
def index():
    # Render the input form (index.html)
    return render_template('index.html', features=breakdown_features)

@app.route('/result', methods=['POST'])
def result():
    # Get input data from the form
    try:
        input_data = [float(request.form[feature]) for feature in breakdown_features]
    except ValueError:
        return "Please enter valid numeric values for all inputs."

    # Create a NumPy array for prediction
    input_array = np.array(input_data).reshape(1, -1)
    # Get breakdown prediction
    breakdown_prediction = breakdown_model.predict(input_array)[0]
    breakdown_result = "Breakdown likely" if breakdown_prediction == 1 else "No breakdown"

    # Combine predictions
    predictions = {
        "Breakdown Prediction": breakdown_result
    }

    # Render the result page
    return render_template('result.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
