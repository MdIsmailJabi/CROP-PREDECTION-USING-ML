"""
Preprocessing Module for Climate-Resilient Crop Prediction
Handles: data cleaning, normalization, encoding, feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os


class CropPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = [
            'Temperature', 'Rainfall', 'Humidity', 'Soil_pH',
            'Nitrogen', 'Phosphorus', 'Potassium', 'Altitude', 'Region'
        ]
        self.is_fitted = False

    def clean_data(self, df):
        """Remove nulls, clip outliers"""
        df = df.dropna().copy()
        # Clip to realistic ranges
        df['Temperature'] = df['Temperature'].clip(-20, 50)
        df['Rainfall'] = df['Rainfall'].clip(0, 5000)
        df['Humidity'] = df['Humidity'].clip(0, 100)
        df['Soil_pH'] = df['Soil_pH'].clip(3.5, 9.5)
        df['Nitrogen'] = df['Nitrogen'].clip(0, 200)
        df['Phosphorus'] = df['Phosphorus'].clip(0, 200)
        df['Potassium'] = df['Potassium'].clip(0, 200)
        df['Altitude'] = df['Altitude'].clip(0, 5000)
        return df

    def fit_transform(self, df):
        """Fit preprocessor and transform training data"""
        df = self.clean_data(df)
        X = df[self.feature_columns].copy()
        y = self.label_encoder.fit_transform(df['Crop'])
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled, y

    def transform(self, input_dict):
        """Transform a single input dict for prediction"""
        X = pd.DataFrame([input_dict])[self.feature_columns]
        return self.scaler.transform(X)

    def decode_label(self, encoded):
        """Decode numeric label back to crop name"""
        return self.label_encoder.inverse_transform(encoded)

    def get_classes(self):
        return self.label_encoder.classes_

    def save(self, path='../models/preprocessor.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path='../models/preprocessor.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
