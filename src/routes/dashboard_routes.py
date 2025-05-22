from flask import Blueprint, render_template, jsonify
import pandas as pd
import numpy as np
from datetime import datetime

dashboard_bp = Blueprint('dashboard', __name__)

def load_dashboard_data():
    # Load the cleaned dataset
    data = pd.read_csv('datasets/cleaned_exped.csv')
    return data

@dashboard_bp.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@dashboard_bp.route('/api/dashboard/stats')
def get_dashboard_stats():
    try:
        data = load_dashboard_data()
        
        # Overall Statistics
        total_expeditions = len(data)
        successful_expeditions = data['success'].sum()
        success_rate = (successful_expeditions / total_expeditions * 100)
        
        # Season-wise Success Rate
        season_stats = data.groupby('season')['success'].agg(['count', 'sum']).reset_index()
        season_stats['success_rate'] = (season_stats['sum'] / season_stats['count'] * 100)
        
        # Peak Difficulty Analysis
        peak_stats = data.groupby('peakid').agg({
            'success': ['count', 'sum'],
            'heightm': 'first'
        }).reset_index()
        peak_stats.columns = ['peakid', 'total_attempts', 'successful_attempts', 'height']
        peak_stats['success_rate'] = (peak_stats['successful_attempts'] / peak_stats['total_attempts'] * 100)
        
        # Risk Assessment (based on team size and oxygen usage)
        risk_stats = data.groupby(['o2used', 'totmembers']).agg({
            'success': ['count', 'sum']
        }).reset_index()
        risk_stats.columns = ['o2used', 'team_size', 'total_attempts', 'successful_attempts']
        risk_stats['success_rate'] = (risk_stats['successful_attempts'] / risk_stats['total_attempts'] * 100)
        
        return jsonify({
            'success': True,
            'data': {
                'overall': {
                    'total_expeditions': int(total_expeditions),
                    'successful_expeditions': int(successful_expeditions),
                    'success_rate': float(success_rate)
                },
                'season_stats': season_stats.to_dict('records'),
                'peak_stats': peak_stats.to_dict('records'),
                'risk_stats': risk_stats.to_dict('records')
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }) 