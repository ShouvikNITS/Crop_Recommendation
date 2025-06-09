# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 23:57:26 2025

@author: Shouvik
"""

"""Weather data fetching and location services"""




import requests
import numpy as np
from geopy.geocoders import Nominatim
from config import Config
class WeatherService:
    """Handles weather data fetching and location services"""

    def __init__(self):
        """Initialize weather service"""
        self.geolocator = Nominatim(user_agent=Config.GEOCODING_USER_AGENT)

    def get_coordinates(self, location):
        """
        Get latitude and longitude for a given location

        Args:
            location (str): Location name (city, state, country)

        Returns:
            tuple: (latitude, longitude) or None if not found
        """
        try:
            location_data = self.geolocator.geocode(location)
            if location_data:
                return location_data.latitude, location_data.longitude
            else:
                print(f"Location '{location}' not found")
                return None
        except Exception as e:
            print(f"Error getting coordinates: {e}")
            return None

    def get_weather_data(self, lat, lon):
        """
        Get historical weather data for the location using NASA POWER API

        Args:
            lat (float): Latitude
            lon (float): Longitude

        Returns:
            dict: Average weather parameters
        """
        try:
            params = {
                'parameters': 'T2M,PRECTOTCORR,RH2M',  # Temperature, Precipitation, Humidity
                'community': 'AG',
                'longitude': lon,
                'latitude': lat,
                'format': 'JSON'
            }

            response = requests.get(
                Config.NASA_POWER_API_URL,
                params=params,
                timeout=Config.WEATHER_API_TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                properties = data.get('properties', {}).get('parameter', {})

                # Extract average values
                temp_data = properties.get('T2M', {})
                precip_data = properties.get('PRECTOTCORR', {})
                humidity_data = properties.get('RH2M', {})

                # Calculate averages
                avg_temp = np.mean(
                    list(temp_data.values())) if temp_data else Config.DEFAULT_VALUES['temperature']
                avg_rainfall = np.mean(list(precip_data.values(
                ))) * 365 if precip_data else Config.DEFAULT_VALUES['rainfall']
                avg_humidity = np.mean(list(humidity_data.values(
                ))) if humidity_data else Config.DEFAULT_VALUES['humidity']

                return {
                    'temperature': avg_temp,
                    'rainfall': avg_rainfall,
                    'humidity': avg_humidity
                }
            else:
                print(f"Weather API error: {response.status_code}")
                return self._get_default_weather()

        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return self._get_default_weather()

    def _get_default_weather(self):
        """Get default weather values when API fails"""
        return {
            'temperature': Config.DEFAULT_VALUES['temperature'],
            'rainfall': Config.DEFAULT_VALUES['rainfall'],
            'humidity': Config.DEFAULT_VALUES['humidity']
        }
