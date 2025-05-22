from flask import Blueprint, request, jsonify, render_template
import pandas as pd
from Machine_learning.model_3.summit_time_model import SummitTimeModel
import os

summit_time_bp = Blueprint('summit_time', __name__)

# Load the trained model at startup
model = SummitTimeModel()
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/model_3'))
model.load_model(model_path)

@summit_time_bp.route('/summit_time', methods=['GET'])
def summit_time_form():
    return render_template('summit_time.html')

@summit_time_bp.route('/predict_summit_time', methods=['POST'])
def predict_summit_time():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Convert oxygen-related boolean values to 'Y'/'N' format
        oxygen_features = ['o2used', 'o2climb', 'o2sleep', 'o2medical']
        for feature in oxygen_features:
            data[feature] = 'Y' if data.get(feature, False) else 'N'

        # Extract features from the input data
        features = {
            'year': int(data.get('year')),
            'season': data.get('season'),
            'totmembers': int(data.get('totmembers')),
            'tothired': int(data.get('tothired')),
            'heightm': float(data.get('heightm')),
            'o2used': data.get('o2used'),
            'o2climb': data.get('o2climb'),
            'o2sleep': data.get('o2sleep'),
            'o2medical': data.get('o2medical'),
            'camps': int(data.get('camps')),
            'rope': float(data.get('rope')),
            'comrte': data.get('comrte'),
            'stdrte': data.get('stdrte')
        }

        # Make prediction
        predicted_days = model.predict(features)
        
        # Get feature importance for explanation
        feature_importance = model.get_feature_importance()
        
        return jsonify({
            'success': True,
            'predicted_days': predicted_days,
            'feature_importance': feature_importance
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500 