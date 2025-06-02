# Dysarthria Severity Classification Pipeline

## Overview
This project provides an end-to-end machine learning pipeline for **dysarthria severity classification** from speech audio. The pipeline supports both advanced (deep learning) and baseline (traditional ML) models, and includes robust feature extraction, data augmentation, and evaluation tools.

> **Note:** This codebase is for **severity-only classification**. Healthy controls are excluded from training and evaluation. The model distinguishes between different levels of dysarthria severity (e.g., mild, moderate, severe).

---

## Features
- **Severity-only classification**: Only dysarthria severity levels are considered (no healthy/control class).
- **Enhanced feature extraction**: Combines Wav2Vec2 embeddings, entropy features, and supplementary acoustic features.
- **Baseline models**: MFCC+SVM and CNN baselines for comparison.
- **Advanced model**: Deep neural network with attention and residual connections.
- **Data augmentation**: For robust model training.
- **Comprehensive evaluation**: Accuracy, precision, recall, F1, confusion matrix, ROC curves, and more.
- **Jupyter notebooks**: End-to-end and baseline pipelines for easy experimentation.

---

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Requirements
- **Audio files**: Place your `.wav` files in a directory (e.g., `audio/`).
- **Metadata CSV**: A file (e.g., `verified_metadata.csv`) with at least two columns:
  - `file_path`: Relative path to each audio file
  - `severity`: Severity label (e.g., 'mild', 'moderate', 'severe')

---

## Usage
### 1. **Train and Evaluate the Enhanced Model**
Run the main pipeline:
```bash
python main.py
```
- Results (metrics, plots, model weights) will be saved in `evaluation_results/`.

### 2. **Jupyter Notebooks**
- `Dysarthria_End2End_Pipeline.ipynb`: Full pipeline for the enhanced model.
- `Dysarthria_Baseline_Pipeline.ipynb`: Baseline models (MFCC+SVM, CNN) pipeline.

### 3. **Custom Training**
- Modify `main.py` or the notebooks to adjust severity classes, model parameters, or feature extraction as needed.

---

## Notes
- **Severity-only setup**: The pipeline automatically filters out healthy controls. Only samples with severity labels (e.g., 'mild', 'moderate', 'severe') are used.
- **Label mapping**: Labels are mapped to consecutive integers for model training.
- **Feature extraction**: Uses Wav2Vec2, entropy, and supplementary features by default.
- **Reproducibility**: Set random seeds as needed for reproducible splits and results.

---

## Contributing & Contact
- Pull requests and issues are welcome!
- For questions or contributions, please contact the maintainer or open an issue on GitHub. 