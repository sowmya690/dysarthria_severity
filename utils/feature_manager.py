import os
import numpy as np
import joblib
from typing import Dict, Tuple, Any, Optional
import logging
import pickle

logger = logging.getLogger(__name__)

class FeatureManager:
    """Manages saving and loading of processed features."""
    
    def __init__(self, feature_dir: str = "processed_features"):
        """
        Initialize feature manager.
        
        Args:
            feature_dir: Directory to store processed features and models
        """
        self.feature_dir = os.path.abspath(feature_dir)
        
        # Create feature directory if it doesn't exist
        try:
            os.makedirs(self.feature_dir, exist_ok=True)
            logger.info(f"Feature directory initialized at: {self.feature_dir}")
        except Exception as e:
            logger.error(f"Failed to create feature directory {self.feature_dir}: {str(e)}")
            raise
        
    def save_features(self, 
                     features: Dict[str, np.ndarray],
                     feature_type: str,
                     split: str) -> None:
        """
        Save processed features to disk.
        
        Args:
            features: Dictionary containing features and metadata
            feature_type: Type of features ('enhanced', 'mfcc', etc.)
            split: Data split ('train', 'val', 'test')
        """
        # Save as joblib file by default (more reliable for mixed data types)
        save_path = os.path.join(self.feature_dir, f"{feature_type}_{split}_features.joblib")
        try:
            joblib.dump(features, save_path)
            logger.info(f"Saved {feature_type} features for {split} split to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save features to {save_path}: {str(e)}")
            raise
    
    def load_features(self, 
                     feature_type: str,
                     split: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load processed features from disk.
        
        Args:
            feature_type: Type of features ('enhanced', 'mfcc', etc.)
            split: Data split ('train', 'val', 'test')
            
        Returns:
            Dictionary containing features and metadata if file exists, None otherwise
        """
        # Try loading from joblib file first
        joblib_path = os.path.join(self.feature_dir, f"{feature_type}_{split}_features.joblib")
        if os.path.exists(joblib_path):
            try:
                features = joblib.load(joblib_path)
                logger.info(f"Loaded {feature_type} features for {split} split from {joblib_path}")
                return features
            except Exception as e:
                logger.error(f"Failed to load features from {joblib_path}: {str(e)}")
        
        # Try loading from legacy .npz file if joblib file doesn't exist
        npz_path = os.path.join(self.feature_dir, f"{feature_type}_{split}_features.npz")
        if os.path.exists(npz_path):
            try:
                with np.load(npz_path, allow_pickle=True) as data:
                    features = {key: data[key] for key in data.files}
                logger.info(f"Loaded {feature_type} features for {split} split from {npz_path}")
                
                # Migrate to joblib format for future use
                self.save_features(features, feature_type, split)
                logger.info(f"Migrated features from {npz_path} to joblib format")
                
                return features
            except Exception as e:
                logger.error(f"Failed to load features from {npz_path}: {str(e)}")
        
        logger.warning(f"No saved features found for {feature_type}_{split}")
        return None
    
    def save_model(self,
                  model_data: Dict[str, Any],
                  feature_type: str) -> None:
        """
        Save feature extraction model (e.g., PCA, scaler).
        
        Args:
            model_data: Dictionary containing model components
            feature_type: Type of features ('enhanced', 'mfcc', etc.)
        """
        save_path = os.path.join(self.feature_dir, f"{feature_type}_model.joblib")
        try:
            joblib.dump(model_data, save_path)
            logger.info(f"Saved {feature_type} model to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model to {save_path}: {str(e)}")
            raise
    
    def load_model(self,
                  feature_type: str) -> Optional[Dict[str, Any]]:
        """
        Load feature extraction model.
        
        Args:
            feature_type: Type of features ('enhanced', 'mfcc', etc.)
            
        Returns:
            Dictionary containing model components if file exists, None otherwise
        """
        # Try loading from joblib file first
        joblib_path = os.path.join(self.feature_dir, f"{feature_type}_model.joblib")
        if os.path.exists(joblib_path):
            try:
                model_data = joblib.load(joblib_path)
                logger.info(f"Loaded {feature_type} model from {joblib_path}")
                return model_data
            except Exception as e:
                logger.error(f"Failed to load model from {joblib_path}: {str(e)}")
        
        # Try loading from legacy .pkl file if joblib file doesn't exist
        pkl_path = os.path.join(self.feature_dir, f"{feature_type}_model.pkl")
        if os.path.exists(pkl_path):
            try:
                model_data = joblib.load(pkl_path)
                logger.info(f"Loaded {feature_type} model from {pkl_path}")
                
                # Migrate to joblib format for future use
                self.save_model(model_data, feature_type)
                logger.info(f"Migrated model from {pkl_path} to joblib format")
                
                return model_data
            except Exception as e:
                logger.error(f"Failed to load model from {pkl_path}: {str(e)}")
        
        logger.warning(f"No saved model found for {feature_type}")
        return None
    
    def features_exist(self,
                      feature_type: str,
                      split: str) -> bool:
        """Check if processed features exist for given type and split."""
        joblib_path = os.path.join(self.feature_dir, f"{feature_type}_{split}_features.joblib")
        npz_path = os.path.join(self.feature_dir, f"{feature_type}_{split}_features.npz")
        return os.path.exists(joblib_path) or os.path.exists(npz_path)
    
    def model_exists(self,
                    feature_type: str) -> bool:
        """Check if feature extraction model exists for given type."""
        joblib_path = os.path.join(self.feature_dir, f"{feature_type}_model.joblib")
        pkl_path = os.path.join(self.feature_dir, f"{feature_type}_model.pkl")
        return os.path.exists(joblib_path) or os.path.exists(pkl_path) 
    
    