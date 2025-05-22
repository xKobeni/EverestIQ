import pandas as pd
import numpy as np
import joblib
import os

class SummitTimeModel:
    def __init__(self):
        self.model = None
        self.label_encoders = None
        self.scaler = None
        
    def load_model(self, model_path):
        """Load the trained model and encoders"""
        try:
            self.model = joblib.load(os.path.join(model_path, 'summit_time_model.joblib'))
            self.label_encoders = joblib.load(os.path.join(model_path, 'label_encoders.joblib'))
            self.scaler = joblib.load(os.path.join(model_path, 'scaler.joblib'))
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_input(self, features):
        """Preprocess input features for prediction"""
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Apply label encoding to categorical features
        categorical_features = ['season', 'o2used', 'o2climb', 'o2sleep', 'o2medical', 'comrte', 'stdrte']
        for feature in categorical_features:
            if feature in self.label_encoders:
                X[feature] = self.label_encoders[feature].transform([X[feature].iloc[0]])
        
        # Scale numerical features
        numerical_features = ['year', 'totmembers', 'tothired', 'heightm', 'camps', 'rope']
        X[numerical_features] = self.scaler.transform(X[numerical_features])
        
        return X
    
    def predict(self, features):
        """Make prediction for summit time"""
        if self.model is None or self.label_encoders is None or self.scaler is None:
            raise ValueError("Model not loaded. Please load the model first.")
        
        # Preprocess input
        X = self.preprocess_input(features)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Round to nearest day
        return round(prediction)
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not loaded. Please load the model first.")
        
        feature_names = [
            'year', 'season', 'totmembers', 'tothired', 'heightm',
            'o2used', 'o2climb', 'o2sleep', 'o2medical', 'camps',
            'rope', 'comrte', 'stdrte'
        ]
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        
        # Sort by importance
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)) 