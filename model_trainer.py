# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 23:58:06 2025

@author: Shouvik
"""

"""Machine learning model training and evaluation"""



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from config import Config
import pickle
class ModelTrainer:
    """Handles model training, evaluation, and persistence"""

    def __init__(self):
        """Initialize model trainer"""
        self.model = None
        self.scaler = None
        self.X_test = None
        self.y_test = None

    def train_model(self, X_train, X_test, y_train, y_test, scaler):
        """
        Train the crop recommendation model

        Args:
            X_train: Training features (scaled)
            X_test: Test features (scaled)
            y_train: Training labels
            y_test: Test labels
            scaler: Fitted scaler object

        Returns:
            float: Model accuracy
        """
        try:
            # Initialize and train Random Forest model
            self.model = RandomForestClassifier(**Config.MODEL_PARAMS)
            self.model.fit(X_train, y_train)

            # Store test data and scaler for evaluation
            self.X_test = X_test
            self.y_test = y_test
            self.scaler = scaler

            # Evaluate model
            y_pred = self.model.predict(X_test)
            #accuracy = accuracy_score(y_test, y_pred)

            #print(f"Model trained successfully!")
            #print(f"Accuracy: {accuracy:.3f}")

            #return accuracy

        except Exception as e:
            raise ValueError(f"Error training model: {e}")

    def save_model(self, model_path=None, scaler_path=None):
        """Save trained model and scaler"""
        model_path = model_path or Config.MODEL_PATH
        scaler_path = scaler_path or Config.SCALER_PATH

        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")

        Config.ensure_model_directory()

        try:
            # Save the model
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)

            # Save the scaler
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)

            print(f"Model saved to {model_path}")
            print(f"Scaler saved to {scaler_path}")
        except Exception as e:
            raise ValueError(f"Error saving model: {e}")

    def load_model(self, model_path=None, scaler_path=None):
        """Load pre-trained model and scaler"""
        model_path = model_path or Config.MODEL_PATH
        scaler_path = scaler_path or Config.SCALER_PATH

        try:
            # Load the model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

            # Load the scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            print("Model and scaler loaded successfully")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found: {e}")
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.model is None:
            raise ValueError("Model not trained.")

        importance_dict = {}
        for feature, importance in zip(Config.FEATURE_COLUMNS, self.model.feature_importances_):
            importance_dict[feature] = importance

        return importance_dict

    def cross_validate(self, X, y, cv_folds=5):
        """
        Perform cross-validation on the model

        Args:
            X: Features
            y: Labels
            cv_folds (int): Number of cross-validation folds

        Returns:
            dict: Cross-validation scores
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        cv_scores = cross_val_score(
            self.model, X, y, cv=cv_folds, scoring='accuracy')

        return {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
