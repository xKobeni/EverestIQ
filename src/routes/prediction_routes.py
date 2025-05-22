from flask import Blueprint, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from Machine_learning.model_2.fatality_risk_model import FatalityRiskModel

prediction_bp = Blueprint('prediction', __name__)

# Load the model and encoders
model = None
label_encoders = {}

def load_model():
    global model, label_encoders
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    models_dir = os.path.join(project_root, 'models', 'model_1')
    
    # Load the model
    model_path = os.path.join(models_dir, 'expedition_success_model.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Please train the model first.")
    model = joblib.load(model_path)
    
    # Load individual encoders
    categorical_features = ['season', 'o2used', 'o2climb', 'o2sleep', 'o2medical', 'comrte', 'stdrte']
    for feature in categorical_features:
        encoder_path = os.path.join(models_dir, f'{feature}_encoder.joblib')
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder for {feature} not found. Please train the model first.")
        label_encoders[feature] = joblib.load(encoder_path)
    
    # Load numerical means
    means_path = os.path.join(models_dir, 'numerical_means.joblib')
    if not os.path.exists(means_path):
        raise FileNotFoundError("Numerical means not found. Please train the model first.")
    numerical_means = joblib.load(means_path)
    
    print("Model and encoders loaded successfully!")

@prediction_bp.route('/prediction', methods=['GET'])
def prediction_form():
    return render_template('index.html')

@prediction_bp.route('/prediction', methods=['POST'])
def predict():
    try:
        if model is None or not label_encoders:
            load_model()
            
        data = request.get_json()
        
        # Prepare input features
        features = {
            'year': int(data['year']),
            'season': data['season'],
            'totmembers': int(data['totmembers']),
            'tothired': int(data['tothired']),
            'heightm': float(data['heightm']),
            'o2used': data['o2used'],
            'o2climb': data['o2climb'],
            'o2sleep': data['o2sleep'],
            'o2medical': data['o2medical'],
            'camps': int(data['camps']),
            'rope': float(data['rope']),
            'comrte': data['comrte'],
            'stdrte': data['stdrte']
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([features])
        
        # Apply label encoding
        categorical_features = ['season', 'o2used', 'o2climb', 'o2sleep', 'o2medical', 'comrte', 'stdrte']
        for feature in categorical_features:
            if feature in label_encoders:
                input_df[feature] = label_encoders[feature].transform([input_df[feature].iloc[0]])
        
        # Make prediction
        prediction = model.predict_proba(input_df)[0]
        success_probability = prediction[1]
        
        return jsonify({
            'success': True,
            'probability': float(success_probability),
            'prediction': 'Likely to succeed' if success_probability > 0.5 else 'Likely to fail'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@prediction_bp.route('/fatality_risk', methods=['GET'])
def fatality_risk_form():
    return render_template('fatality_risk.html') 