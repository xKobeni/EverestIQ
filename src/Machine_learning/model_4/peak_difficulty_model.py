import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

class PeakDifficultyModel:
    def __init__(self):
        self.cluster_model = None
        self.classifier = None
        self.scaler = None
        self.feature_columns = [
            'success_rate',
            'average_team_size',
            'fatality_rate',
            'average_time_to_summit',
            'total_expeditions',
            'oxygen_usage_rate',
            'commercial_route_rate'
        ]
        
    def prepare_data(self, exped_df, peaks_df):
        """Prepare and aggregate data by peak"""
        # Merge expedition data with peaks data
        merged_df = pd.merge(exped_df, peaks_df[['peakid', 'heightm']], on='peakid', how='left')
        
        # Convert columns to appropriate types and handle missing values
        numeric_columns = ['success1', 'totmembers', 'mdeaths', 'o2used', 'comrte']
        for col in numeric_columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
            # Fill missing values with 0 for binary columns
            if col in ['success1', 'o2used', 'comrte']:
                merged_df[col] = merged_df[col].fillna(0)
            # Fill missing values with median for continuous columns
            else:
                merged_df[col] = merged_df[col].fillna(merged_df[col].median())
        
        # Calculate summit time if not already present
        if 'summit_time' not in merged_df.columns:
            merged_df['summit_time'] = (pd.to_datetime(merged_df['smtdate'], errors='coerce') - 
                                      pd.to_datetime(merged_df['bcdate'], errors='coerce')).dt.days
            # Fill missing summit times with median
            merged_df['summit_time'] = merged_df['summit_time'].fillna(merged_df['summit_time'].median())
        
        # Calculate aggregated features by peak
        peak_stats = merged_df.groupby('peakid').agg({
            'success1': ['mean', 'count'],  # success rate and total expeditions
            'totmembers': 'mean',  # average team size
            'mdeaths': lambda x: (x > 0).mean(),  # fatality rate
            'summit_time': 'mean',  # average time to summit
            'o2used': 'mean',  # oxygen usage rate
            'comrte': 'mean'  # commercial route rate
        }).reset_index()
        
        # Rename columns
        peak_stats.columns = ['peakid', 'success_rate', 'total_expeditions', 
                            'average_team_size', 'fatality_rate', 
                            'average_time_to_summit', 'oxygen_usage_rate',
                            'commercial_route_rate']
        
        # Remove peaks with too few expeditions (less than 5)
        peak_stats = peak_stats[peak_stats['total_expeditions'] >= 5]
        
        # Handle any remaining missing values in numeric columns
        numeric_columns = peak_stats.columns.difference(['peakid'])
        for col in numeric_columns:
            if peak_stats[col].isna().any():
                peak_stats[col] = peak_stats[col].fillna(peak_stats[col].median())
        
        # Ensure all numeric columns are float
        peak_stats[numeric_columns] = peak_stats[numeric_columns].astype(float)
        
        # Verify no NaN values remain
        assert not peak_stats[numeric_columns].isna().any().any(), "NaN values still present in the data"
        
        return peak_stats
    
    def create_initial_clusters(self, X):
        """Create initial difficulty clusters using K-means"""
        # Scale the features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Create DataFrame with scaled features
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        # Perform K-means clustering
        self.cluster_model = KMeans(n_clusters=3, random_state=42)
        cluster_labels = self.cluster_model.fit_predict(X_scaled)
        
        # Map cluster labels to difficulty levels
        # We'll sort clusters by their average success rate to determine difficulty
        cluster_means = X_scaled_df.groupby(cluster_labels).mean()
        
        # Find the cluster with highest success rate (Easy)
        easy_cluster = cluster_means['success_rate'].idxmax()
        # Find the cluster with lowest success rate (Hard)
        hard_cluster = cluster_means['success_rate'].idxmin()
        # The remaining cluster is Moderate
        moderate_cluster = list(set(range(3)) - {easy_cluster, hard_cluster})[0]
        
        # Create difficulty mapping
        difficulty_mapping = {
            easy_cluster: 'Easy',
            moderate_cluster: 'Moderate',
            hard_cluster: 'Hard'
        }
        
        return np.array([difficulty_mapping[label] for label in cluster_labels])
    
    def train(self, X, y=None):
        """Train the model using both clustering and classification"""
        if y is None:
            # If no labels provided, create them using clustering
            y = self.create_initial_clusters(X)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = self.classifier.predict(X_test_scaled)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return self.classifier.score(X_test_scaled, y_test)
    
    def predict(self, X):
        """Predict difficulty level for new peaks"""
        if self.classifier is None or self.scaler is None:
            raise ValueError("Model has not been trained yet!")
        
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get probability estimates for each difficulty level"""
        if self.classifier is None or self.scaler is None:
            raise ValueError("Model has not been trained yet!")
        
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)
    
    def save_model(self, path='models/peak_difficulty_model.joblib'):
        """Save the trained model and scaler"""
        if self.classifier is None or self.scaler is None:
            raise ValueError("No model to save!")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model components
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path='models/peak_difficulty_model.joblib'):
        """Load a trained model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        
        model_data = joblib.load(path)
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns'] 