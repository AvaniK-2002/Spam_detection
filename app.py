from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('spam_detector.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    is_number_saved = int(request.form['is_number_saved'])
    message_type = request.form['message_type']
    
    # Ensure message_type is formatted correctly
    if message_type.lower() not in ['sms', 'email']:
        return render_template('index.html', prediction_text="Invalid message type. Please enter 'sms' or 'email'.")
    
    # Transform the input data into a DataFrame
    input_data = pd.DataFrame([[message, is_number_saved, message_type]], 
                              columns=['message', 'is_number_saved', 'message_type'])
    
    # Perform prediction
    prediction = model.predict(input_data)[0]
    result = 'Spam' if prediction == 1 else 'Not Spam'
    
    return render_template('index.html', prediction_text=f'This message is: {result}')

if __name__ == "__main__":
    app.run(debug=True)
