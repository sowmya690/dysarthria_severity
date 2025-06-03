import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import librosa
from typing import Tuple, Optional


class MFCCFeatureExtractor:
    """Extracts MFCC features from audio signals."""
    
    def __init__(self, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.scaler = StandardScaler()
        
    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract MFCC features from an audio file."""
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, 
                                    n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Transpose to get time as first dimension
        mfccs = mfccs.T
        
        # Take mean across time dimension
        return np.mean(mfccs, axis=0)
    
    def fit_transform(self, audio_paths: list) -> np.ndarray:
        """Extract and standardize features from multiple audio files."""
        features = np.array([self.extract_features(path) for path in audio_paths])
        return self.scaler.fit_transform(features)
    
    def transform(self, audio_paths: list) -> np.ndarray:
        """Transform new audio files using fitted scaler."""
        features = np.array([self.extract_features(path) for path in audio_paths])
        return self.scaler.transform(features)


class MFCCSVMClassifier:
    """SVM classifier using MFCC features."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0):
        self.feature_extractor = MFCCFeatureExtractor()
        self.classifier = SVC(kernel=kernel, C=C, probability=True)
        
    def fit(self, audio_paths: list, labels: np.ndarray) -> None:
        """Train the SVM classifier."""
        features = self.feature_extractor.fit_transform(audio_paths)
        self.classifier.fit(features, labels)
        
    def predict(self, audio_paths: list) -> np.ndarray:
        """Make predictions on new audio files."""
        features = self.feature_extractor.transform(audio_paths)
        return self.classifier.predict(features)
    
    def predict_proba(self, audio_paths: list) -> np.ndarray:
        """Get probability estimates for predictions."""
        features = self.feature_extractor.transform(audio_paths)
        return self.classifier.predict_proba(features)


class CNNClassifier(nn.Module):
    """1D CNN classifier for audio classification."""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 4):
        super(CNNClassifier, self).__init__()
        
        # Using 1D convolutions for sequence data
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Input shape: [batch, channels, features]
        
        # Convolutional layers with pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def train_cnn_model(model: nn.Module,
                   train_loader: torch.utils.data.DataLoader,
                   val_loader: torch.utils.data.DataLoader,
                   num_epochs: int = 50,
                   lr: float = 0.001,
                   device: Optional[torch.device] = None) -> Tuple[nn.Module, list, list]:
    """Train the CNN model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        train_losses.append(running_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f} - Val Loss: {val_losses[-1]:.4f}")
        
    return model, train_losses, val_losses 