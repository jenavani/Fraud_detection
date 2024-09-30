import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model = joblib.load('catboost_model.joblib')
    scaler = joblib.load('standard_scaler.joblib')
    feature_names = joblib.load('feature_names.joblib')
    logger.info("Model, scaler, and feature names loaded successfully")
except Exception as e:
    logger.error(f"Error loading model, scaler, or feature names: {str(e)}")
    raise

def prepare_features(data):
    features = {
        'amount': float(data['amount']),
        'oldbalanceOrg': float(data['oldbalanceOrg']),
        'newbalanceOrig': float(data['newbalanceOrg']),
        'oldbalanceDest': float(data['oldbalanceDest']),
        'newbalanceDest': float(data['newbalanceDest'])
    }
    
    # Derived features
    features['balance_diff_orig'] = features['oldbalanceOrg'] - features['newbalanceOrig']
    features['balance_diff_dest'] = features['oldbalanceDest'] - features['newbalanceDest']
    features['balance_ratio_orig'] = features['oldbalanceOrg'] / (features['newbalanceOrig'] + 1)
    features['balance_ratio_dest'] = features['oldbalanceDest'] / (features['newbalanceDest'] + 1)
    
    # One-hot encode the transaction type
    for feature in feature_names:
        if feature.startswith('type_'):
            features[feature] = 1 if data['transaction_type'] == feature.split('_')[1] else 0
    
    # Ensure all features are present and in the correct order
    return pd.DataFrame([{name: features.get(name, 0) for name in feature_names}])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        logger.info(f"Received data: {data}")
        
        feature_df = prepare_features(data)
        logger.info(f"Prepared features: {feature_df.to_dict(orient='records')}")
        
        scaled_data = scaler.transform(feature_df)
        logger.info(f"Scaled data shape: {scaled_data.shape}")
        
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)
        
        result = 'Fraud' if prediction[0] == 1 else 'Legitimate'
        
        response = {
            'result': result,
            'fraud_probability': float(prediction_proba[0][1]),
            'debug_info': {
                'input_data': feature_df.to_dict(orient='records')[0],
                'scaled_data': scaled_data.tolist()[0],
                'prediction': prediction.tolist()[0],
                'prediction_probability': prediction_proba.tolist()[0]
            }
        }
        
        logger.info(f"Prediction result: {response}")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)