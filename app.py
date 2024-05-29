from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load the scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from the form
    features = ['temp', 'dew', 'humidity', 'windgust', 'windspeed', 'cloudcover', 'visibility']
    feature_values = [float(request.form.get(feature)) for feature in features]
    
    # Standardize the input values
    input_data_scaled = scaler.transform(np.array(feature_values).reshape(1, -1))
    
    # Predict the class based on the standardized input
    predicted_class = model.predict(input_data_scaled)
    
    # Decode the predicted class
    predicted_class = label_encoder.inverse_transform(predicted_class)[0]
    
    return render_template("result.html", predicted_class=predicted_class)

if __name__ == "__main__":
    app.run(debug=True)
