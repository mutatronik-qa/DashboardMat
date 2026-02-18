"""Machine learning models for EMG signal classification.

This module provides implementations of various classifiers for EMG signal
classification, including Random Forest and Support Vector Machines.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm

class EMGClassifier:
    """Base class for EMG classifiers."""
    
    def __init__(self, model_type: str = 'rf', **kwargs):
        """Initialize the classifier.
        
        Args:
            model_type (str, optional): Type of classifier. Options: 'rf' (Random Forest),
                                      'svm' (Support Vector Machine). Defaults to 'rf'.
            **kwargs: Additional arguments to pass to the underlying classifier.
        """
        self.model_type = model_type.lower()
        self.model = self._init_model(model_type, **kwargs)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.classes_ = None
        
    def _init_model(self, model_type: str, **kwargs) -> Any:
        """Initialize the underlying classifier model.
        
        Args:
            model_type (str): Type of classifier.
            **kwargs: Additional arguments for the classifier.
            
        Returns:
            Any: Initialized classifier instance.
            
        Raises:
            ValueError: If an unknown model type is specified.
        """
        if model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        elif model_type == 'svm':
            return SVC(
                C=kwargs.get('C', 1.0),
                kernel=kwargs.get('kernel', 'rbf'),
                gamma=kwargs.get('gamma', 'scale'),
                probability=True,
                random_state=kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> 'EMGClassifier':
        """Train the classifier on the given data.
        
        Args:
            X (np.ndarray): Training data (n_samples, n_features).
            y (np.ndarray): Target labels (n_samples,).
            feature_names (Optional[List[str]], optional): Names of the features.
                                                         Defaults to None.
                                                         
        Returns:
            EMGClassifier: The fitted classifier.
        """
        # Store feature names if provided
        self.feature_names = feature_names
        
        # Store class labels
        self.classes_ = np.unique(y)
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the model
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for the given data.
        
        Args:
            X (np.ndarray): Input data (n_samples, n_features).
            
        Returns:
            np.ndarray: Predicted class labels (n_samples,).
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the given data.
        
        Args:
            X (np.ndarray): Input data (n_samples, n_features).
            
        Returns:
            np.ndarray: Class probabilities (n_samples, n_classes).
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the classifier on the given test data.
        
        Args:
            X (np.ndarray): Test data (n_samples, n_features).
            y (np.ndarray): True labels (n_samples,).
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        y_pred = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, np.ndarray]:
        """Perform k-fold cross-validation.
        
        Args:
            X (np.ndarray): Input data (n_samples, n_features).
            y (np.ndarray): Target labels (n_samples,).
            cv (int, optional): Number of folds. Defaults to 5.
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of cross-validation scores.
        """
        # Create a pipeline with scaling and the model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', self.model)
        ])
        
        # Define metrics to compute
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        # Perform cross-validation
        cv_scores = {}
        for metric in scoring:
            scores = cross_val_score(
                pipeline, X, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                scoring=metric, n_jobs=-1
            )
            cv_scores[metric] = scores
        
        return cv_scores
    
    def grid_search(self, X: np.ndarray, y: np.ndarray, 
                   param_grid: Dict[str, List[Any]], cv: int = 5) -> Dict[str, Any]:
        """Perform grid search for hyperparameter tuning.
        
        Args:
            X (np.ndarray): Training data (n_samples, n_features).
            y (np.ndarray): Target labels (n_samples,).
            param_grid (Dict[str, List[Any]]): Parameter grid for grid search.
            cv (int, optional): Number of folds. Defaults to 5.
            
        Returns:
            Dict[str, Any]: Dictionary with grid search results.
        """
        # Create a pipeline with scaling and the model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', self.model)
        ])
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Update the model with the best estimator
        self.model = grid_search.best_estimator_.named_steps['classifier']
        self.scaler = grid_search.best_estimator_.named_steps['scaler']
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances.
        
        Returns:
            np.ndarray: Feature importances.
            
        Raises:
            ValueError: If the model does not support feature importances.
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.mean(np.abs(self.model.coef_), axis=0)
        else:
            raise ValueError("Model does not support feature importances")
    
    def save(self, filepath: str) -> None:
        """Save the model to a file.
        
        Args:
            filepath (str): Path to save the model.
            
        Raises:
            ValueError: If the file extension is not .pkl or .joblib.
        """
        if not filepath.endswith('.pkl') and not filepath.endswith('.joblib'):
            raise ValueError("File extension must be .pkl or .joblib")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save the model
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model_type': self.model_type,
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'classes_': self.classes_
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'EMGClassifier':
        """Load a model from a file.
        
        Args:
            filepath (str): Path to the saved model.
            
        Returns:
            EMGClassifier: The loaded model.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not .pkl or .joblib.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        if not (filepath.endswith('.pkl') or filepath.endswith('.joblib')):
            raise ValueError("File extension must be .pkl or .joblib")
        
        # Load the model
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create a new instance
        classifier = cls(model_type=data['model_type'])
        classifier.model = data['model']
        classifier.scaler = data['scaler']
        classifier.feature_names = data['feature_names']
        classifier.classes_ = data['classes_']
        
        return classifier


def train_model(X_train: np.ndarray, y_train: np.ndarray, 
                model_type: str = 'rf', **kwargs) -> EMGClassifier:
    """Train a classifier on the given data.
    
    Args:
        X_train (np.ndarray): Training data (n_samples, n_features).
        y_train (np.ndarray): Training labels (n_samples,).
        model_type (str, optional): Type of classifier. Options: 'rf' (Random Forest),
                                  'svm' (Support Vector Machine). Defaults to 'rf'.
        **kwargs: Additional arguments for the classifier.
        
    Returns:
        EMGClassifier: The trained classifier.
    """
    classifier = EMGClassifier(model_type=model_type, **kwargs)
    classifier.fit(X_train, y_train)
    return classifier


def evaluate_model(classifier: EMGClassifier, X_test: np.ndarray, 
                  y_test: np.ndarray) -> Dict[str, Any]:
    """Evaluate a classifier on the given test data.
    
    Args:
        classifier (EMGClassifier): The trained classifier.
        X_test (np.ndarray): Test data (n_samples, n_features).
        y_test (np.ndarray): True labels (n_samples,).
        
    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics.
    """
    return classifier.evaluate(X_test, y_test)


def save_model(classifier: EMGClassifier, filepath: str) -> None:
    """Save a classifier to a file.
    
    Args:
        classifier (EMGClassifier): The classifier to save.
        filepath (str): Path to save the model.
    """
    classifier.save(filepath)


def load_model(filepath: str) -> EMGClassifier:
    """Load a classifier from a file.
    
    Args:
        filepath (str): Path to the saved model.
        
    Returns:
        EMGClassifier: The loaded classifier.
    """
    return EMGClassifier.load(filepath)