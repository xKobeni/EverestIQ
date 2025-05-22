from flask import Blueprint, request, jsonify, render_template
import pandas as pd
from Machine_learning.model_2.fatality_risk_model import FatalityRiskModel
import joblib
import os

fatality_risk_bp = Blueprint('fatality_risk', __name__)

# Load the trained model at startup
model = FatalityRiskModel()
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/fatality_risk_model.joblib'))
model.load_model(model_path)

@fatality_risk_bp.route('/fatality_risk', methods=['GET'])
def fatality_risk_form():
    return render_template('fatality_risk.html')

@fatality_risk_bp.route('/predict_fatality_risk', methods=['POST'])
def predict_fatality_risk():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    # Extract features from the input data
    totmembers = float(data.get('totmembers', 0))
    tothired = float(data.get('tothired', 0))
    total_team_size = totmembers + tothired
    team_size_ratio = totmembers / total_team_size if total_team_size > 0 else 0

    features = {
        'totmembers': totmembers,
        'tothired': tothired,
        'o2used': data.get('o2used'),
        'o2climb': data.get('o2climb'),
        'o2sleep': data.get('o2sleep'),
        'o2medical': data.get('o2medical'),
        'season': data.get('season'),
        'peakid': data.get('peakid'),
        'route1': data.get('route1'),
        'success1': data.get('success1'),
        'year': data.get('year'),
        'heightm': data.get('heightm'),
        'camps': data.get('camps'),
        'rope': data.get('rope'),
        'comrte': data.get('comrte'),
        'stdrte': data.get('stdrte'),
        'team_size_ratio': team_size_ratio,
        'total_team_size': total_team_size
    }

    # Convert features to a DataFrame
    X = pd.DataFrame([features])

    # Make prediction
    prediction = model.predict(X)[0]
    return jsonify({'predicted_risk_level': prediction}) 