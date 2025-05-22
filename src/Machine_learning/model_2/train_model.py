# NOTE: The model expects the following columns in exped.csv:
# totmembers, tothired, o2used, season, peakid, route1, success1, mdeaths, hdeaths
import pandas as pd
import numpy as np
from fatality_risk_model import FatalityRiskModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def plot_feature_importance(model):
    """Plot feature importance"""
    importance_df = model.get_feature_importance()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(15), x='importance', y='feature')
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Dynamically resolve the paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    exped_path = os.path.abspath(os.path.join(base_dir, '../../../datasets/exped.csv'))
    peaks_path = os.path.abspath(os.path.join(base_dir, '../../../datasets/peaks.csv'))
    
    # Load the data
    print("Loading data...")
    exped_df = pd.read_csv(exped_path)
    peaks_df = pd.read_csv(peaks_path)
    
    # Initialize the model
    model = FatalityRiskModel()
    
    # Prepare the data
    print("Preparing data...")
    X, y = model.prepare_data(exped_df, peaks_df)
    
    # Print class distribution before SMOTE
    print("\nClass distribution BEFORE SMOTE:")
    print(y.value_counts(normalize=True))

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train the model with SMOTE
    print("Training the fatality risk model with SMOTE...")
    model.train(X_train, y_train, use_smote=True)

    # Print class distribution after SMOTE
    print("\nClass distribution AFTER SMOTE:")
    print(Counter(y_train))
    
    # Evaluate the model
    print("\nEvaluating model performance...")
    cv_scores = model.evaluate(X, y)
    print(f"\nCross-validation scores:")
    print(f"Mean F1 Score: {cv_scores['mean_f1']:.3f} (+/- {cv_scores['std_f1']:.3f})")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print detailed model performance
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix plot...")
    plot_confusion_matrix(y_test, y_pred, labels=y.unique())
    
    # Plot feature importance
    print("\nGenerating feature importance plot...")
    plot_feature_importance(model)
    
    # Save the model
    model_path = os.path.abspath(os.path.join(base_dir, '../../../models/fatality_risk_model.joblib'))
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main() 