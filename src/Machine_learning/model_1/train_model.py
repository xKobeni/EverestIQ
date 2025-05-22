import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import sys

# Add the src directory to the Python path so that preprocess.py can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from preprocess import clean_expedition_data

def train_and_save_model():
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(project_root, 'models', 'model_1')
    os.makedirs(models_dir, exist_ok=True)

    # Define paths for raw and cleaned data
    exped_path = os.path.join(project_root, 'datasets', 'exped.csv')
    peaks_path = os.path.join(project_root, 'datasets', 'peaks.csv')
    cleaned_output_path = os.path.join(project_root, 'datasets', 'cleaned_exped.csv')
    
    # Clean the data using the preprocess module
    clean_expedition_data(exped_path, peaks_path, cleaned_output_path)
    
    # Load the cleaned dataset
    data = pd.read_csv(cleaned_output_path)
    
    # Print dataset statistics
    total_expeditions = len(data)
    successful_expeditions = data['success'].sum()
    success_rate = (successful_expeditions / total_expeditions) * 100
    print("\nDataset Statistics:")
    print(f"Total number of expeditions: {total_expeditions}")
    print(f"Number of successful expeditions: {successful_expeditions}")
    print(f"Success rate: {success_rate:.2f}%")

    # Select relevant features for prediction
    features = [
        'year', 'season', 'totmembers', 'tothired', 'heightm',
        'o2used', 'o2climb', 'o2sleep', 'o2medical',
        'camps', 'rope', 'comrte', 'stdrte'
    ]

    # Print unique values in success1 before mapping
    print('Unique values in success1 before mapping:', data['success1'].unique())

    # Create target variable (success) from boolean True/False
    data['success'] = data['success1'].map({True: 1, False: 0})

    # Drop rows where target is NaN
    before_drop = len(data)
    data = data.dropna(subset=['success'])
    after_drop = len(data)
    print(f"Dropped {before_drop - after_drop} rows with NaN target values.")

    # Prepare features
    X = data[features].copy()

    # Handle categorical variables
    categorical_features = ['season', 'o2used', 'o2climb', 'o2sleep', 'o2medical', 'comrte', 'stdrte']
    label_encoders = {}
    
    for feature in categorical_features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature].fillna('N'))
        label_encoders[feature] = le
        # Save label encoders
        joblib.dump(le, os.path.join(models_dir, f'{feature}_encoder.joblib'))

    # Fill missing values
    X = X.fillna(X.mean())

    # Save mean values for numerical features
    numerical_means = X.mean()
    joblib.dump(numerical_means, os.path.join(models_dir, 'numerical_means.joblib'))

    # Prepare target variable
    y = data['success']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print model performance
    print("\nModel Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, os.path.join(models_dir, 'expedition_success_model.joblib'))

    # Save feature names
    joblib.dump(features, os.path.join(models_dir, 'feature_names.joblib'))

    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    feature_importance.to_csv(os.path.join(models_dir, 'feature_importance.csv'), index=False)

    print(f"\nModel and preprocessing components saved to {models_dir}")

if __name__ == "__main__":
    train_and_save_model() 