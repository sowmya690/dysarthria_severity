import os
import numpy as np
import joblib
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.signal
import scipy.stats
import librosa
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractorWithPCA:
    def __init__(self, n_pca_components: int = 20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca_components)
        self.n_pca_components = n_pca_components

    def extract_wav2vec_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        min_required_length = 4000  # Minimum required length in samples for Wav2Vec2
        if len(audio) < min_required_length:
            pad_width = min_required_length - len(audio)
            audio = np.pad(audio, (0, pad_width), mode="constant")
            logger.info(f"Padded short audio from {len(audio) - pad_width} to {len(audio)} samples.")

        input_values = self.processor(audio, sampling_rate=sr, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            embeddings = self.model(input_values).last_hidden_state.squeeze(0).cpu().numpy()
        logger.debug(f"Wav2Vec2 embeddings shape: {embeddings.shape}")
        return embeddings

    def compute_entropy_features(self, audio, sr, frame_size_ms=50, frame_shift_ms=25, num_filters=5):
        logger.debug(f"compute_entropy_features called with audio length: {len(audio)}, sr: {sr}")

        frame_size = int(sr * frame_size_ms / 1000)
        frame_shift = int(sr * frame_shift_ms / 1000)

        if len(audio) < frame_size:
            pad_width = frame_size - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant')
            logger.info(f"Padded short audio from {len(audio) - pad_width} to {len(audio)} samples.")

        logger.debug(f"Frame size (samples): {frame_size}, Frame shift: {frame_shift}")
        freqs = np.linspace(100, 4000, num_filters)
        logger.debug(f"Gabor filter center frequencies: {freqs}")

        t = np.arange(-frame_size // 2, frame_size // 2)
        gabor_filters = []
        for f in freqs:
            sigma = frame_size / 8
            gabor = np.exp(-0.5 * (t / sigma) ** 2) * np.cos(2 * np.pi * f * t / sr)
            gabor_filters.append(gabor)

        gabor_filters = np.array(gabor_filters)
        logger.debug(f"Generated {len(gabor_filters)} Gabor filters")

        num_frames = 1 + (len(audio) - frame_size) // frame_shift
        if num_frames <= 0:
            logger.warning(f"Insufficient audio for entropy feature extraction. Returning empty array.")
            return np.zeros((1, num_filters))

        frames = np.zeros((num_frames, frame_size))
        for i in range(num_frames):
            start = i * frame_shift
            end = start + frame_size
            frames[i] = audio[start:end]

        logger.debug(f"Total frames extracted: {num_frames}")
        entropy_features = []
        for idx, frame in enumerate(frames):
            frame_entropies = []
            for gabor in gabor_filters:
                filtered = np.convolve(frame, gabor, mode='same')
                energy = filtered ** 2
                energy /= np.sum(energy) + 1e-10
                entropy = -np.sum(energy * np.log2(energy + 1e-10))
                frame_entropies.append(entropy)
            entropy_features.append(frame_entropies)
            if idx < 3 or idx == num_frames - 1:
                logger.debug(f"Frame {idx} entropy: {frame_entropies}")

        entropy_features = np.array(entropy_features)
        logger.debug(f"Final entropy features shape: {entropy_features.shape}")
        return entropy_features

    def extract_and_process(self, all_frames: list, sr: int, fit_pca: bool = True):
        wav2vec_features = []
        entropy_features = []

        for i, frames in enumerate(all_frames):
            concatenated = np.concatenate(frames)
            logger.info(f"Processing utterance {i+1}/{len(all_frames)} with {len(concatenated)} samples")

            w2v_feat = self.extract_wav2vec_features(concatenated, sr)
            entropy_feat = self.compute_entropy_features(concatenated, sr)

            wav2vec_features.append(np.mean(w2v_feat, axis=0))
            entropy_features.append(np.mean(entropy_feat, axis=0))

        logger.info(f"Total utterances processed: {len(wav2vec_features)}")

        # Convert to numpy arrays
        wav2vec_features = np.array(wav2vec_features)
        entropy_features = np.array(entropy_features)

        if fit_pca:
            standardized = self.scaler.fit_transform(wav2vec_features)
            wav2vec_pca = self.pca.fit_transform(standardized)
            logger.info(f"Fitted PCA and StandardScaler on wav2vec features")
        else:
            standardized = self.scaler.transform(wav2vec_features)
            wav2vec_pca = self.pca.transform(standardized)
            logger.info(f"Transformed wav2vec features using existing PCA model")

        # Concatenate wav2vec and entropy features
        combined_features = np.hstack([wav2vec_pca, entropy_features])
        logger.info(f"Combined features shape: {combined_features.shape}")
        
        return combined_features

    def save_pca_model(self, path: str):
        joblib.dump((self.scaler, self.pca), path)
        logger.info(f"Saved PCA model to {path}")

    def load_pca_model(self, path: str):
        self.scaler, self.pca = joblib.load(path)
        logger.info(f"Loaded PCA model from {path}")
    
    def save_entropy_features(self, features: np.ndarray, path: str):
        np.save(path, features)
        logger.info(f"Entropy features saved to {path}")

    def load_entropy_features(self, path: str) -> np.ndarray:
        logger.info(f"Loading entropy features from {path}")
        return np.load(path)

class FeatureProcessor:
    def fit_transform(self, wav2vec_feats: np.ndarray):
        return wav2vec_feats

    def fuse_features(self, wav2vec_feats: np.ndarray, entropy_feats: np.ndarray) -> np.ndarray:
        min_len = min(len(wav2vec_feats), len(entropy_feats))
        fused = np.concatenate([wav2vec_feats[:min_len], entropy_feats[:min_len]], axis=1)
        logger.info(f"Fused features shape: {fused.shape}")
        return fused
