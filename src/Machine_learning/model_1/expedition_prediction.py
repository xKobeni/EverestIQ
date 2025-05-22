import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
expeditions = pd.read_csv('../../../datasets/exped.csv')
peaks = pd.read_csv('../../../datasets/peaks.csv')

# Merge expeditions with peaks data to get peak information
data = pd.merge(expeditions, peaks[['peakid', 'heightm', 'region']], on='peakid', how='left')

# Select relevant features for prediction
features = [
    'year', 'season', 'totmembers', 'tothired', 'heightm',
    'o2used', 'o2climb', 'o2sleep', 'o2medical',
    'camps', 'rope', 'comrte', 'stdrte'
]

# Create target variable (success)
data['success'] = data['success1'].map({'Y': 1, 'N': 0})

# Prepare features
X = data[features].copy()

# Handle categorical variables
categorical_features = ['season', 'o2used', 'o2climb', 'o2sleep', 'o2medical', 'comrte', 'stdrte']
for feature in categorical_features:
    le = LabelEncoder()
    X[feature] = le.fit_transform(X[feature].fillna('N'))

# Fill missing values
X = X.fillna(X.mean())

# Prepare target variable
y = data['success']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print model performance
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance for Expedition Success Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nFeature Importance:")
print(feature_importance) 