import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from typing import List, Tuple, Optional
import os
import pandas as pd


# NOTE: This module is for dysarthria severity classification only. Healthy controls are excluded from training and evaluation.

class MFCCDataset(Dataset):
    """Dataset class for MFCC features."""
    
    def __init__(self, audio_paths: List[str], labels: List[int], 
                 n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512):
        self.audio_paths = audio_paths
        self.labels = labels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load and extract MFCCs
        y, sr = librosa.load(self.audio_paths[idx], sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, 
                                   n_mfcc=self.n_mfcc,
                                   n_fft=self.n_fft,
                                   hop_length=self.hop_length)
        
        # Convert to tensor and add channel dimension
        mfccs = torch.FloatTensor(mfccs).unsqueeze(0)  # Shape: [1, n_mfcc, time]
        return mfccs, self.labels[idx]


def load_metadata(data_dir: str) -> pd.DataFrame:
    """Load and process metadata from the dataset directory."""
    metadata_path = os.path.join(data_dir, "verified_metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    severity_classes = ['mild', 'moderate', 'severe']  # Update as needed
    df = df[df['severity'].isin(severity_classes)]
    label_map = {label: idx for idx, label in enumerate(severity_classes)}
    df['label'] = df['severity'].map(label_map)
    print('Label mapping:', label_map)
    return df


def prepare_data(data_dir: str, 
                batch_size: int = 32,
                test_size: float = 0.2,
                val_size: float = 0.125,
                random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], List[str], List[str]]:
    """Prepare data loaders for training, validation, and testing."""
    
    # Load metadata
    df = load_metadata(data_dir)
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    # First split into train+val and test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    
    # Then split train+val into train and val
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_state, stratify=train_val_df['label']
    )
    
    # Create datasets
    train_dataset = MFCCDataset(
        train_df['path'].tolist(),
        train_df['label'].tolist()
    )
    
    val_dataset = MFCCDataset(
        val_df['path'].tolist(),
        val_df['label'].tolist()
    )
    
    test_dataset = MFCCDataset(
        test_df['path'].tolist(),
        test_df['label'].tolist()
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return (
        train_loader,
        val_loader,
        test_loader,
        train_df['path'].tolist(),
        val_df['path'].tolist(),
        test_df['path'].tolist()
    )


def extract_mfcc_features(audio_paths: List[str],
                         n_mfcc: int = 13,
                         n_fft: int = 2048,
                         hop_length: int = 512) -> np.ndarray:
    """Extract MFCC features from a list of audio files."""
    features = []
    for path in audio_paths:
        y, sr = librosa.load(path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr,
                                   n_mfcc=n_mfcc,
                                   n_fft=n_fft,
                                   hop_length=hop_length)
        # Take mean across time dimension
        features.append(np.mean(mfccs, axis=1))
    return np.array(features) 