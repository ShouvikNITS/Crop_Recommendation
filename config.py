# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 23:56:18 2025

@author: Shouvik
"""

import os


class Config:
    """Configuration class for the crop recommendation system"""

    # Model parameters
    MODEL_PARAMS = {
        'n_estimators': 100,
        'random_state': 42,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }

    # Feature columns
    FEATURE_COLUMNS = ['N', 'P', 'K',
                       'temperature', 'humidity', 'ph', 'rainfall']

    # Default values
    DEFAULT_VALUES = {
        'nitrogen': 60.0,
        'phosphorus': 25.0,
        'potassium': 40.0,
        'ph': 6.5,
        'temperature': 25.0,
        'humidity': 60.0,
        'rainfall': 1000.0
    }

    # Model file paths
    MODEL_PATH = 'models/crop_model.pkl'
    SCALER_PATH = 'models/scaler.pkl'

    # API settings
    WEATHER_API_TIMEOUT = 30
    GEOCODING_USER_AGENT = "crop_recommender"
    NASA_POWER_API_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"

    # Ensure models directory exists
    @staticmethod
    def ensure_model_directory():
        """Create models directory if it doesn't exist"""
        os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
