"""
Dataset Generator for Climate-Resilient Crop Prediction
Simulates realistic agro-climatic data for hilly and polar regions
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_crop_dataset(n_samples=2000):
    """
    Generate synthetic but realistic agro-climatic dataset
    Features: Temperature, Rainfall, Humidity, Soil pH, N, P, K, Altitude, Region
    Target: Crop (Rice, Ragi, Jowar, Corn, Wheat)
    """

    crops = ['Rice', 'Ragi', 'Jowar', 'Corn', 'Wheat']
    data = []

    # Crop-specific environmental profiles
    crop_profiles = {
        'Rice': {
            'temp': (22, 35), 'rainfall': (1200, 2500), 'humidity': (60, 90),
            'ph': (5.5, 7.0), 'N': (80, 120), 'P': (40, 80), 'K': (40, 80),
            'altitude': (0, 600), 'region': 'Hilly'
        },
        'Ragi': {
            'temp': (15, 30), 'rainfall': (500, 1200), 'humidity': (40, 70),
            'ph': (5.0, 7.5), 'N': (40, 80), 'P': (30, 60), 'K': (30, 60),
            'altitude': (600, 1800), 'region': 'Hilly'
        },
        'Jowar': {
            'temp': (20, 35), 'rainfall': (400, 1000), 'humidity': (30, 65),
            'ph': (6.0, 8.0), 'N': (30, 70), 'P': (20, 50), 'K': (20, 50),
            'altitude': (0, 500), 'region': 'Polar'
        },
        'Corn': {
            'temp': (18, 32), 'rainfall': (600, 1500), 'humidity': (50, 80),
            'ph': (5.5, 7.5), 'N': (60, 100), 'P': (40, 70), 'K': (40, 70),
            'altitude': (300, 1500), 'region': 'Hilly'
        },
        'Wheat': {
            'temp': (5, 22), 'rainfall': (200, 800), 'humidity': (25, 60),
            'ph': (6.0, 7.5), 'N': (50, 90), 'P': (30, 65), 'K': (30, 65),
            'altitude': (800, 3000), 'region': 'Polar'
        }
    }

    samples_per_crop = n_samples // len(crops)

    for crop, profile in crop_profiles.items():
        for _ in range(samples_per_crop):
            noise = np.random.normal(0, 0.08)
            temp = np.random.uniform(*profile['temp']) + noise * 5
            rainfall = np.random.uniform(*profile['rainfall']) + noise * 200
            humidity = np.random.uniform(*profile['humidity']) + noise * 10
            soil_ph = np.random.uniform(*profile['ph']) + noise * 0.3
            N = np.random.uniform(*profile['N']) + noise * 15
            P = np.random.uniform(*profile['P']) + noise * 10
            K = np.random.uniform(*profile['K']) + noise * 10
            altitude = np.random.uniform(*profile['altitude']) + noise * 100
            region = profile['region']

            data.append({
                'Temperature': round(max(-5, temp), 2),
                'Rainfall': round(max(50, rainfall), 2),
                'Humidity': round(max(10, min(100, humidity)), 2),
                'Soil_pH': round(max(4.0, min(9.0, soil_ph)), 2),
                'Nitrogen': round(max(5, N), 2),
                'Phosphorus': round(max(5, P), 2),
                'Potassium': round(max(5, K), 2),
                'Altitude': round(max(0, altitude), 2),
                'Region': 1 if region == 'Hilly' else 0,  # 1=Hilly, 0=Polar
                'Crop': crop
            })

    df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def generate_market_data():
    """Market price and demand data per crop"""
    market_data = {
        'Crop': ['Rice', 'Ragi', 'Jowar', 'Corn', 'Wheat'],
        'Avg_Market_Price_Per_Quintal': [2183, 3846, 2015, 1760, 2275],
        'Demand_Level': ['High', 'Medium', 'Low', 'Medium', 'High'],
        'Export_Potential': ['High', 'Medium', 'Low', 'High', 'High'],
        'Profit_Margin_Percent': [28, 35, 18, 30, 25],
        'Growing_Season_Days': [120, 100, 110, 90, 130],
        'Water_Requirement_mm': [1200, 450, 400, 600, 400]
    }
    return pd.DataFrame(market_data)


if __name__ == '__main__':
    os.makedirs('../data', exist_ok=True)
    df = generate_crop_dataset(2000)
    df.to_csv('../data/crop_data.csv', index=False)
    market = generate_market_data()
    market.to_csv('../data/market_data.csv', index=False)
    print(f"Generated {len(df)} crop samples")
    print(df['Crop'].value_counts())
