# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 23:59:15 2025

@author: Shouvik
"""

"""Crop prediction service"""




import numpy as np
from weather_service import WeatherService
from config import Config
class CropPredictor:
    """Handles crop prediction based on location and soil parameters"""

    def __init__(self, model, scaler):
        """
        Initialize predictor

        Args:
            model: Trained ML model
            scaler: Fitted scaler
        """
        self.model = model
        self.scaler = scaler
        self.weather_service = WeatherService()

    def predict_crops(self, location, n=None, p=None, k=None, ph=None, top_k=3):
        """
        Predict top crops for a given location

        Args:
            location (str): Location name
            n (float, optional): Nitrogen content
            p (float, optional): Phosphorus content
            k (float, optional): Potassium content
            ph (float, optional): Soil pH
            top_k (int): Number of top predictions to return

        Returns:
            list: Top crop recommendations with probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

        # Get coordinates
        coords = self.weather_service.get_coordinates(location)
        if coords is None:
            return []

        lat, lon = coords
        print(f"Location: {location}")
        print(f"Coordinates: {lat:.4f}, {lon:.4f}")

        # Get weather data
        weather_data = self.weather_service.get_weather_data(lat, lon)

        # Use provided values or defaults
        n = n or Config.DEFAULT_VALUES['nitrogen']
        p = p or Config.DEFAULT_VALUES['phosphorus']
        k = k or Config.DEFAULT_VALUES['potassium']
        ph = ph or Config.DEFAULT_VALUES['ph']

        print(f"Using NPK values: N={n}, P={p}, K={k}, pH={ph}")

        # Prepare input features
        input_features = np.array([[
            n, p, k,
            weather_data['temperature'],
            weather_data['humidity'],
            ph,
            weather_data['rainfall']
        ]])

        # Scale features
        input_scaled = self.scaler.transform(input_features)

        # Get predictions and probabilities
        probabilities = self.model.predict_proba(input_scaled)[0]

        # Get all classes and their probabilities
        classes = self.model.classes_
        crop_probs = list(zip(classes, probabilities))

        # Sort by probability and get top k
        crop_probs.sort(key=lambda x: x[1], reverse=True)
        top_crops = crop_probs[:top_k]

        return top_crops

    def predict_single_sample(self, features, true_label=None):
        """
        Predict for a single sample with detailed output

        Args:
            features (list): [N, P, K, temp, humidity, ph, rainfall]
            true_label (str, optional): True crop label for accuracy check

        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded.")

        # Prepare input
        input_features = np.array([features])
        input_scaled = self.scaler.transform(input_features)

        # Make prediction
        prediction = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]

        # Get probability for the predicted class
        classes = self.model.classes_
        pred_prob = probabilities[np.where(classes == prediction)[0][0]]

        results = {
            'predicted': prediction,
            'confidence': pred_prob,
            'all_probabilities': dict(zip(classes, probabilities))
        }

        if true_label:
            results['actual'] = true_label
            results['correct'] = prediction == true_label

        return results
