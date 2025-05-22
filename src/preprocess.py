import pandas as pd
import numpy as np
import os

def clean_expedition_data(exped_path, peaks_path, output_path):
    # Load datasets
    expeditions = pd.read_csv(exped_path, low_memory=False)
    peaks = pd.read_csv(peaks_path)

    # Merge with peak info
    data = pd.merge(expeditions, peaks[['peakid', 'heightm', 'region']], on='peakid', how='left')

    # Remove expeditions with missing essential info
    essential_cols = ['year', 'season', 'totmembers', 'tothired', 'heightm',
                      'o2used', 'o2climb', 'o2sleep', 'o2medical',
                      'camps', 'rope', 'comrte', 'stdrte', 'success1']
    data = data.dropna(subset=essential_cols, how='any')

    # Standardize categorical values (fill NAs with 'N' for binary Y/N fields)
    yn_fields = ['o2used', 'o2climb', 'o2sleep', 'o2medical', 'comrte', 'stdrte']
    for col in yn_fields:
        data[col] = data[col].fillna('N').replace({True: 'Y', False: 'N'})

    # Standardize season
    data['season'] = data['season'].fillna('Unknown')

    # Standardize numeric fields
    num_fields = ['year', 'totmembers', 'tothired', 'heightm', 'camps', 'rope']
    for col in num_fields:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna(subset=num_fields, how='any')

    # Standardize target
    data['success'] = data['success1'].map({True: 1, False: 0, 'Y': 1, 'N': 0})
    data = data.dropna(subset=['success'])

    # Save cleaned data
    data.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}. Shape: {data.shape}")

if __name__ == '__main__':
    exped_path = os.path.join('datasets', 'exped.csv')
    peaks_path = os.path.join('datasets', 'peaks.csv')
    output_path = os.path.join('datasets', 'cleaned_exped.csv')
    clean_expedition_data(exped_path, peaks_path, output_path) 