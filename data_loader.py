# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 23:56:46 2025

@author: Shouvik
"""

"""Data loading and preprocessing utilities"""




import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import Config
class DataLoader:
    """Handles data loading and preprocessing for crop recommendation"""

    def __init__(self, csv_file_path):
        """
        Initialize data loader

        Args:
            csv_file_path (str): Path to the CSV file containing crop data
        """
        self.csv_file_path = csv_file_path
        self.df = None
        self.scaler = StandardScaler()

    def load_data(self):
        """Load CSV data and perform basic validation"""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            self._validate_data()
            return self.df
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")

    def _validate_data(self):
        """Validate that required columns exist in the dataset"""
        required_cols = Config.FEATURE_COLUMNS + ['label']
        missing_cols = [
            col for col in required_cols if col not in self.df.columns]

        if missing_cols:
            # Add missing columns with default values if possible
            if 'ph' in missing_cols and 'ph' not in self.df.columns:
                self.df['ph'] = Config.DEFAULT_VALUES['ph']
                missing_cols.remove('ph')

            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

    def prepare_data_for_training(self, test_size=0.2, random_state=42):
        """
        Prepare data for model training

        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random state for reproducibility

        Returns:
            tuple: (X_train_scaled, X_test_scaled, y_train, y_test)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        X = self.df[Config.FEATURE_COLUMNS]
        y = self.df['label']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def get_scaler(self):
        """Get the fitted scaler"""
        return self.scaler

    def get_feature_info(self):
        """Get information about the dataset features"""
        if self.df is None:
            raise ValueError("Data not loaded.")

        return {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'unique_crops': self.df['label'].nunique(),
            'crop_names': sorted(self.df['label'].unique()),
            'feature_stats': self.df[Config.FEATURE_COLUMNS].describe()
        }
