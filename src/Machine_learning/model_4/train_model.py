import pandas as pd
import os
from peak_difficulty_model import PeakDifficultyModel

def load_data():
    """Load and prepare the expedition and peaks data"""
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '..'))
    
    # Load the datasets
    exped_df = pd.read_csv(os.path.join(project_root, 'datasets', 'cleaned_exped.csv'), low_memory=False)
    peaks_df = pd.read_csv(os.path.join(project_root, 'datasets', 'peaks.csv'))
    
    # Calculate summit time in days
    exped_df['summit_time'] = (pd.to_datetime(exped_df['smtdate']) - pd.to_datetime(exped_df['bcdate'])).dt.days
    
    return exped_df, peaks_df

def train_peak_difficulty_model():
    """Train the peak difficulty classification model"""
    # Get the absolute path to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '..'))
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(project_root, 'models', 'model_4')
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    exped_df, peaks_df = load_data()
    
    # Initialize and train the model
    model = PeakDifficultyModel()
    
    # Prepare the data
    peak_stats = model.prepare_data(exped_df, peaks_df)
    
    # Select features for training
    X = peak_stats[model.feature_columns]
    
    # Train the model
    print("Training Peak Difficulty Model...")
    accuracy = model.train(X)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    # Save the model
    model_path = os.path.join(models_dir, 'peak_difficulty_model.joblib')
    model.save_model(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Print some example predictions
    print("\nExample Predictions:")
    sample_peaks = peak_stats.sample(5)
    predictions = model.predict(sample_peaks[model.feature_columns])
    probabilities = model.predict_proba(sample_peaks[model.feature_columns])
    
    for i, (_, peak) in enumerate(sample_peaks.iterrows()):
        print(f"\nPeak ID: {peak['peakid']}")
        print(f"Success Rate: {peak['success_rate']:.2%}")
        print(f"Fatality Rate: {peak['fatality_rate']:.2%}")
        print(f"Predicted Difficulty: {predictions[i]}")
        print("Difficulty Probabilities:")
        for j, difficulty in enumerate(['Easy', 'Moderate', 'Hard']):
            print(f"  {difficulty}: {probabilities[i][j]:.2%}")

if __name__ == "__main__":
    train_peak_difficulty_model() 