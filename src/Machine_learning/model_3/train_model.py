import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def load_and_preprocess_data():
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '..'))
    
    # Load the expedition data using absolute path
    df = pd.read_csv(os.path.join(project_root, 'datasets', 'cleaned_exped.csv'), low_memory=False)
    
    # Calculate summit time in days using the correct column names
    df['summit_time'] = (pd.to_datetime(df['smtdate']) - pd.to_datetime(df['bcdate'])).dt.days
    
    # Remove rows where summit time is negative or unreasonably high (e.g., > 100 days)
    df = df[(df['summit_time'] > 0) & (df['summit_time'] <= 100)]
    
    # Select features for the model
    features = [
        'year', 'season', 'totmembers', 'tothired', 'heightm',
        'o2used', 'o2climb', 'o2sleep', 'o2medical', 'camps',
        'rope', 'comrte', 'stdrte'
    ]
    
    X = df[features].copy()
    y = df['summit_time']
    
    # Handle categorical features
    categorical_features = ['season', 'o2used', 'o2climb', 'o2sleep', 'o2medical', 'comrte', 'stdrte']
    label_encoders = {}
    
    for feature in categorical_features:
        label_encoders[feature] = LabelEncoder()
        X[feature] = label_encoders[feature].fit_transform(X[feature].astype(str))
    
    # Scale numerical features
    numerical_features = ['year', 'totmembers', 'tothired', 'heightm', 'camps', 'rope']
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    return X, y, label_encoders, scaler

def evaluate_model(y_true, y_pred):
    """Calculate and print various evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print("\nModel Performance Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f} days")
    print(f"Root Mean Square Error (RMSE): {rmse:.2f} days")
    print(f"R² Score: {r2:.4f}")
    
    # Calculate percentage of predictions within different error ranges
    error_ranges = [1, 2, 3, 5, 7]
    for days in error_ranges:
        within_range = np.mean(np.abs(y_true - y_pred) <= days) * 100
        print(f"Predictions within ±{days} days: {within_range:.1f}%")

def train_model():
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '..'))
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(project_root, 'models', 'model_3')
    os.makedirs(models_dir, exist_ok=True)
    
    # Load and preprocess data
    X, y, label_encoders, scaler = load_and_preprocess_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Print training set performance
    print("\nTraining Set Performance:")
    evaluate_model(y_train, y_train_pred)
    
    # Print test set performance
    print("\nTest Set Performance:")
    evaluate_model(y_test, y_test_pred)
    
    # Print feature importance
    feature_names = [
        'year', 'season', 'totmembers', 'tothired', 'heightm',
        'o2used', 'o2climb', 'o2sleep', 'o2medical', 'camps',
        'rope', 'comrte', 'stdrte'
    ]
    importance = model.feature_importances_
    feature_importance = dict(zip(feature_names, importance))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    print("\nFeature Importance:")
    for feature, importance in sorted_importance.items():
        print(f"{feature}: {importance:.4f}")
    
    # Save the model and encoders
    joblib.dump(model, os.path.join(models_dir, 'summit_time_model.joblib'))
    joblib.dump(label_encoders, os.path.join(models_dir, 'label_encoders.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    
    print("\nModel and encoders saved successfully!")

if __name__ == "__main__":
    train_model() 