# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 23:59:52 2025

@author: Shouvik
"""

"""Main crop recommendation system that orchestrates all components"""




from data_loader import DataLoader
from model_trainer import ModelTrainer
from crop_predictor import CropPredictor
from config import Config
class CropRecommendationSystem:
    """Main system that orchestrates all components"""

    def __init__(self, csv_file_path):
        """
        Initialize the crop recommendation system

        Args:
            csv_file_path (str): Path to the CSV file containing crop data
        """
        self.csv_file_path = csv_file_path
        self.data_loader = DataLoader(csv_file_path)
        self.model_trainer = ModelTrainer()
        self.crop_predictor = None
        self.model_evaluator = None

    def setup_system(self, train_new_model=True):
        """
        Set up the complete system

        Args:
            train_new_model (bool): Whether to train a new model or load existing
        """
        # Load data
        print("Loading data...")
        self.data_loader.load_data()

        if train_new_model:
            self._train_new_model()
        else:
            self._load_existing_model()

        # Initialize predictor
        self.crop_predictor = CropPredictor(
            self.model_trainer.model,
            self.model_trainer.scaler
        )

        print("System setup complete!")

    def _train_new_model(self):
        """Train a new model"""
        print("Preparing data for training...")
        X_train, X_test, y_train, y_test = self.data_loader.prepare_data_for_training()

        print("Training model...")
        self.model_trainer.train_model(
            X_train, X_test, y_train, y_test,
            self.data_loader.get_scaler()
        )

        # Initialize evaluator
        #self.model_evaluator = ModelEvaluator(
            #self.model_trainer.model, X_test, y_test
        #)

        # Save model
        print("Saving model...")
        self.model_trainer.save_model()

    def _load_existing_model(self):
        """Load existing model"""
        print("Loading existing model...")
        try:
            self.model_trainer.load_model()
        except FileNotFoundError:
            print("No existing model found. Training new model...")
            self._train_new_model()

    def get_recommendations(self, location, n=None, p=None, k=None, ph=None, top_k=3):
        """
        Get crop recommendations for a location

        Args:
            location (str): Location name
            n (float, optional): Nitrogen content
            p (float, optional): Phosphorus content  
            k (float, optional): Potassium content
            ph (float, optional): Soil pH
            top_k (int): Number of recommendations

        Returns:
            list: Top crop recommendations
        """
        if self.crop_predictor is None:
            raise ValueError("System not set up. Call setup_system() first.")

        return self.crop_predictor.predict_crops(location, n, p, k, ph, top_k)

    def evaluate_model(self, detailed=True, plot_confusion_matrix=False):
        """Evaluate the trained model"""
        if self.model_evaluator is None:
            raise ValueError(
                "Model evaluator not available. Train a new model first.")

        return self.model_evaluator.evaluate_model(detailed, plot_confusion_matrix)

    def get_model_info(self):
        """Get information about the model and data"""
        info = {
            'data_info': self.data_loader.get_feature_info(),
            'feature_importance': self.model_trainer.get_feature_importance() if self.model_trainer.model else None
        }
        return info
