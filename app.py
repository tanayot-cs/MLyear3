import sys
import joblib
import pandas as pd
from flask import Flask, render_template, request

# Load models
rf_model = joblib.load('C:/Users/ACER-444/Downloads/random_forest_model.pkl')
xgb_model = joblib.load('C:/Users/ACER-444/Downloads/xgboost_model.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        global_active_power = float(request.form['global_active_power'])
        voltage = float(request.form['voltage'])
        global_intensity = float(request.form['global_intensity'])
        model_selected = request.form['model']

        # Prepare input data
        input_data = pd.DataFrame({
            'Global_active_power': [global_active_power],
            'Voltage': [voltage],
            'Global_intensity': [global_intensity]
        })

        # Make prediction
        if model_selected == 'Random Forest':
            prediction = rf_model.predict(input_data)
        else:
            prediction = xgb_model.predict(input_data)

        # Prepare result
        recommendation = "Your energy usage is within normal limits."
        if prediction[0] > 10:  # Example threshold for high usage
            recommendation = "Consider reducing usage or upgrading to more energy-efficient appliances."

        return f"<h1>Prediction Result</h1><p>Predicted Sub Metering 3 Usage: {prediction[0]:.2f} kWh</p><p>{recommendation}</p>"

    except Exception as e:
        return f"<h1>Error</h1><p>{str(e)}</p>"

if __name__ == '__main__':
    app.run(debug=True)
