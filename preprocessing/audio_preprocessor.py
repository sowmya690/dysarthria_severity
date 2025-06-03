import os
import numpy as np
import soundfile as sf
import resampy
import webrtcvad
from typing import Tuple, List

class AudioPreprocessor:
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self.vad = webrtcvad.Vad(3)  # Aggressiveness mode 3

    def resample_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        # Read audio file using soundfile
        audio, sr = sf.read(audio_path)
        if sr != self.target_sr:
            audio = resampy.resample(audio, sr, self.target_sr)
            sr = self.target_sr
        return audio, sr

    def remove_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio_int16 = (audio * 32768).astype(np.int16)
        frame_duration = 30  # ms
        frame_size = int(sr * frame_duration / 1000)
        frames = [audio_int16[i:i + frame_size] for i in range(0, len(audio_int16), frame_size)]

        voiced_frames = [frame for frame in frames if len(frame) == frame_size and self.vad.is_speech(frame.tobytes(), sr)]
        if not voiced_frames:
            # No voiced frames detected, return original
            return audio

        return np.concatenate(voiced_frames).astype(np.float32) / 32768.0

    def segment_frames(self, audio: np.ndarray, sr: int, frame_duration: float = 0.05) -> List[np.ndarray]:
        frame_size = int(sr * frame_duration)
        return [audio[i:i + frame_size] for i in range(0, len(audio), frame_size)]

    def process_audio(self, audio_path: str) -> List[np.ndarray]:
        if not os.path.isfile(audio_path):
            raise ValueError(f"File not found: {audio_path}")

        try:
            audio, sr = self.resample_audio(audio_path)
            audio = self.remove_silence(audio, sr)
        except Exception as e:
            print(f"[ERROR] Failed to load/process audio '{audio_path}': {e}")
            raise
        
        frames = self.segment_frames(audio, sr)
        return frames
