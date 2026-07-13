import pickle
from flask import Flask, render_template, request
import numpy as np
import traceback
import os

app = Flask(__name__)

# Load trained model securely using absolute paths for Serverless (Vercel) compatibility
try:
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    model = pickle.load(open(model_path, 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        features = [float(x) for x in request.form.values()]
        
        # Convert user-friendly Years to Days as expected by the model
        features[9] = features[9] * 365   # DAYS_BIRTH
        features[10] = features[10] * 365 # DAYS_EMPLOYED
        
        # Convert input into array
        final_input = [np.array(features)]
        
        # Predict result
        prediction = model.predict(final_input)
        
        # Display output
        if prediction[0] == 1:
            result = "Credit Card Approved"
        else:
            result = "Credit Card Rejected"
            
        return render_template('result.html', prediction_text=result)
    except Exception as e:
        print(f"Prediction Error: {e}")
        traceback.print_exc()
        return render_template('result.html', prediction_text="Error in prediction. Please check your inputs.")

if __name__ == "__main__":
    app.run(debug=True)
