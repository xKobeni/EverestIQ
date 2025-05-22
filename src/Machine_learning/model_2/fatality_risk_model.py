import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
import joblib
import os

class FatalityRiskModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_columns = [
            'totmembers',  # team size
            'tothired',    # hired staff
            'o2used',      # oxygen used
            'season',
            'peakid',
            'route1',
            'success1',
            'year',        # added year
            'heightm',     # added peak height
            'camps',       # added number of camps
            'rope',        # added fixed rope length
            'comrte',      # added commercial route flag
            'stdrte',      # added standard route flag
            'team_size_ratio',
            'total_team_size'
        ]
        
    def prepare_data(self, exped_df, peaks_df):
        # Merge with peaks data to get height information
        exped_df = pd.merge(exped_df, peaks_df[['peakid', 'heightm']], on='peakid', how='left')
        
        # Create target variable based on fatality counts
        exped_df['fatality_count'] = exped_df['mdeaths'] + exped_df['hdeaths']
        
        # Create risk levels with more balanced categories
        exped_df['risk_level'] = pd.cut(
            exped_df['fatality_count'],
            bins=[-1, 0, 1, float('inf')],
            labels=['Low', 'Medium', 'High']
        )
        
        # Create interaction features
        exped_df['team_size_ratio'] = exped_df['totmembers'] / (exped_df['totmembers'] + exped_df['tothired'])
        exped_df['total_team_size'] = exped_df['totmembers'] + exped_df['tothired']
        
        # Handle missing values
        exped_df['camps'] = exped_df['camps'].fillna(0)
        exped_df['rope'] = exped_df['rope'].fillna(0)
        exped_df['heightm'] = exped_df['heightm'].fillna(exped_df['heightm'].median())
        
        # Select features and target
        X = exped_df[self.feature_columns]
        y = exped_df['risk_level']
        
        return X, y
    
    def create_preprocessor(self):
        # Define categorical and numerical features
        categorical_features = ['season', 'peakid', 'route1', 'o2used', 'success1', 'comrte', 'stdrte']
        numerical_features = ['totmembers', 'tothired', 'year', 'heightm', 'camps', 'rope', 
                            'team_size_ratio', 'total_team_size']
        
        # Create preprocessing steps
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())  # More robust to outliers
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
    
    def train(self, X, y, use_smote=False):
        self.create_preprocessor()
        class_counts = y.value_counts()
        class_weights = {cls: len(y) / (len(class_counts) * count) 
                        for cls, count in class_counts.items()}

        # Preprocess X to numeric before SMOTE
        X_numeric = self.preprocessor.fit_transform(X)
        y_array = np.array(y)

        # Optionally apply SMOTE for class balancing
        if use_smote:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_numeric, y_array)
        else:
            X_resampled, y_resampled = X_numeric, y_array

        # Fit classifier on resampled data
        classifier = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8
        )
        classifier.fit(X_resampled, y_resampled)

        # Save the pipeline for prediction
        self.model = make_pipeline(self.preprocessor, classifier)

        # Get feature importance
        feature_names = self.preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y):
        """Evaluate model performance with cross-validation"""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1_weighted')
        return {
            'mean_f1': scores.mean(),
            'std_f1': scores.std()
        }
    
    def get_feature_importance(self):
        """Return feature importance analysis"""
        if not hasattr(self, 'feature_importance'):
            raise ValueError("Model has not been trained yet!")
        return self.feature_importance
    
    def save_model(self, path='models/fatality_risk_model.joblib'):
        if self.model is None:
            raise ValueError("No model to save!")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model and feature importance
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path='models/fatality_risk_model.joblib'):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_importance = model_data['feature_importance'] 