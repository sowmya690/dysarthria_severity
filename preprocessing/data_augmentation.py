import numpy as np
import librosa
import torch
import torchaudio
from typing import Tuple, Optional
import random

class AudioAugmenter:
    """Audio augmentation techniques for dysarthria speech data."""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        
    def add_noise(self, audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
        """Add random white noise to the audio."""
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise
    
    def time_stretch(self, audio: np.ndarray, rate_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Time stretch the audio by a random rate."""
        rate = random.uniform(*rate_range)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio: np.ndarray, 
                   n_steps_range: Tuple[float, float] = (-2, 2)) -> np.ndarray:
        """Pitch shift the audio by a random number of steps."""
        n_steps = random.uniform(*n_steps_range)
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
    
    def speed_perturb(self, audio: np.ndarray, 
                     speed_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Speed perturbation of the audio."""
        speed = random.uniform(*speed_range)
        return librosa.effects.time_stretch(audio, rate=speed)
    
    def add_reverb(self, audio: np.ndarray, room_scale: float = 0.5) -> np.ndarray:
        """Add reverb effect to the audio."""
        # Simple reverb simulation using convolution
        reverb_length = int(self.sr * room_scale)
        reverb = np.exp(-np.arange(reverb_length) / (self.sr * room_scale))
        reverb = reverb / np.sum(reverb)
        return np.convolve(audio, reverb, mode='same')
    
    def apply_augmentation(self, audio: np.ndarray, 
                          augmentations: Optional[list] = None) -> np.ndarray:
        """Apply a random combination of augmentations to the audio."""
        if augmentations is None:
            augmentations = [
                self.add_noise,
                self.time_stretch,
                self.pitch_shift,
                self.speed_perturb,
                self.add_reverb
            ]
        
        # Randomly select 1-3 augmentations
        num_augmentations = random.randint(1, 3)
        selected_augmentations = random.sample(augmentations, num_augmentations)
        
        augmented_audio = audio.copy()
        for aug_func in selected_augmentations:
            augmented_audio = aug_func(augmented_audio)
        
        return augmented_audio

class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset class that applies augmentation during training."""
    
    def __init__(self, audio_paths: list, labels: list, 
                 augmenter: AudioAugmenter, augment_prob: float = 0.5):
        self.audio_paths = audio_paths
        self.labels = labels
        self.augmenter = augmenter
        self.augment_prob = augment_prob
        
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load audio
        audio, sr = librosa.load(self.audio_paths[idx], sr=self.augmenter.sr)
        
        # Apply augmentation with probability
        if random.random() < self.augment_prob:
            audio = self.augmenter.apply_augmentation(audio)
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio)
        
        return audio_tensor, self.labels[idx]

def create_augmented_dataloader(audio_paths: list, labels: list,
                              batch_size: int = 32,
                              augment_prob: float = 0.5,
                              num_workers: int = 4) -> torch.utils.data.DataLoader:
    """Create a DataLoader with augmented data."""
    augmenter = AudioAugmenter()
    dataset = AugmentedDataset(audio_paths, labels, augmenter, augment_prob)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    ) 