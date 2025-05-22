from flask import Blueprint, request, jsonify, render_template
import pandas as pd
import os
from Machine_learning.model_4.peak_difficulty_model import PeakDifficultyModel

peak_difficulty_bp = Blueprint('peak_difficulty', __name__)

# Initialize the model
model = None

def load_model():
    global model
    if model is None:
        model = PeakDifficultyModel()
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        model_path = os.path.join(project_root, 'models', 'model_4', 'peak_difficulty_model.joblib')
        try:
            model.load_model(model_path)
        except FileNotFoundError:
            print(f"Warning: Model file not found at {model_path}. Please train the model first.")

@peak_difficulty_bp.route('/peak-difficulty', methods=['GET'])
def peak_difficulty_form():
    """Render the peak difficulty prediction form"""
    return render_template('peak_difficulty.html')

@peak_difficulty_bp.route('/peak-difficulty/predict', methods=['POST'])
def predict_difficulty():
    """Predict difficulty level for a peak"""
    try:
        if model is None:
            load_model()
        
        data = request.get_json()
        
        # Prepare input features
        features = {
            'success_rate': float(data['success_rate']),
            'average_team_size': int(data['average_team_size']),
            'fatality_rate': float(data['fatality_rate']),
            'average_time_to_summit': float(data['average_time_to_summit']),
            'total_expeditions': int(data['total_expeditions']),
            'oxygen_usage_rate': float(data['oxygen_usage_rate']),
            'commercial_route_rate': float(data['commercial_route_rate'])
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([features])
        
        # Make prediction
        difficulty = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        
        # Format probabilities
        prob_dict = {
            'Easy': float(probabilities[0]),
            'Moderate': float(probabilities[1]),
            'Hard': float(probabilities[2])
        }
        
        return jsonify({
            'success': True,
            'difficulty': difficulty,
            'probabilities': prob_dict
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@peak_difficulty_bp.route('/peak-difficulty/peaks', methods=['GET'])
def get_peaks():
    """Get list of peaks with their difficulty ratings"""
    try:
        if model is None:
            load_model()
        
        # Load peaks data
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        peaks_df = pd.read_csv(os.path.join(project_root, 'datasets', 'peaks.csv'))
        
        # Load expedition data for feature calculation
        exped_df = pd.read_csv(os.path.join(project_root, 'datasets', 'cleaned_exped.csv'), low_memory=False)
        
        # Calculate features for each peak
        peak_stats = model.prepare_data(exped_df, peaks_df)
        
        # Make predictions
        X = peak_stats[model.feature_columns]
        difficulties = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Combine results
        results = []
        for i, (_, peak) in enumerate(peak_stats.iterrows()):
            peak_info = peaks_df[peaks_df['peakid'] == peak['peakid']].iloc[0]
            results.append({
                'peak_id': str(peak['peakid']),  # Convert to string
                'peak_name': str(peak_info['pkname']),  # Use pkname instead of peakname
                'height': float(peak_info['heightm']),
                'difficulty': str(difficulties[i]),
                'success_rate': float(peak['success_rate']),
                'fatality_rate': float(peak['fatality_rate']),
                'total_expeditions': int(peak['total_expeditions']),
                'probabilities': {
                    'Easy': float(probabilities[i][0]),
                    'Moderate': float(probabilities[i][1]),
                    'Hard': float(probabilities[i][2])
                }
            })
        
        return jsonify({
            'success': True,
            'peaks': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }) 