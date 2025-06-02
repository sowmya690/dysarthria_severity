# NOTE: This script is for dysarthria severity classification only. Healthy controls are excluded from training and evaluation.
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from preprocessing.feature_extractor import FeatureExtractorWithPCA
from models.enhanced_model import EnhancedDysarthriaClassifier, train_enhanced_model
from models.baseline_models import MFCCSVMClassifier, CNNClassifier
from evaluation.baseline_evaluation import BaselineModelEvaluator
from utils.feature_manager import FeatureManager
import logging

logger = logging.getLogger(__name__)


def extract_features(feature_extractor_type: str, 
                    paths: list, 
                    sr: int, 
                    feature_manager: FeatureManager,
                    split: str,
                    y: np.ndarray,
                    **kwargs) -> dict:
    """
    Extract features using specified feature extractor.
    
    Args:
        feature_extractor_type: Type of feature extractor ('enhanced' or 'pca')
        paths: List of audio file paths
        sr: Sample rate
        feature_manager: Feature manager instance
        split: Data split ('train', 'val', 'test')
        y: Labels
        **kwargs: Additional arguments for feature extractors
        
    Returns:
        Dictionary containing extracted features and labels
    """
    if feature_manager.features_exist(feature_extractor_type, split):
        print(f"[DEBUG] Loading saved {feature_extractor_type} features for {split} split...")
        return feature_manager.load_features(feature_extractor_type, split)
    
    print(f"[DEBUG] Extracting {feature_extractor_type} features for {split} split...")
    all_frames = [[librosa.load(path, sr=None)[0]] for path in paths]
    
    extractor = FeatureExtractorWithPCA(**kwargs)
    
    # For val/test splits, load the fitted models from training
    if split in ['val', 'test']:
        if not feature_manager.model_exists(feature_extractor_type):
            raise ValueError(
                f"Cannot process {split} split without trained models. "
                "Please process training split first."
            )
        model_data = feature_manager.load_model(feature_extractor_type)
        if model_data is None:
            raise ValueError("Failed to load trained models")
        
        extractor.scaler = model_data['scaler']
        extractor.pca = model_data['pca']
        print(f"[DEBUG] Loaded pre-fitted models for {split} split")
    
    # Extract and process features
    features = extractor.extract_and_process(all_frames, sr, fit_pca=(split == 'train'))
    print(f"[DEBUG] Extracted features shape: {features.shape}")
    
    feature_dict = {
        'features': features,
        'labels': y
    }
    
    feature_manager.save_features(feature_dict, feature_extractor_type, split)
    
    # Save fitted models after processing training data
    if split == 'train':
        feature_manager.save_model({
            'scaler': extractor.scaler,
            'pca': extractor.pca
        }, feature_extractor_type)
        print("[DEBUG] Saved fitted models after training split")
    
    return feature_dict


def tune_hyperparameters(train_loader, val_loader, device, input_dim, num_classes):
    """
    Perform hyperparameter tuning using grid search.
    Returns the best parameters and best validation accuracy.
    """
    param_grid = {
        'n_pca_components': [20],
        'learning_rates': [0.001, 0.0005, 0.0001],
        'weight_decays': [1e-4, 1e-3, 1e-5],
        'hidden_dims_configs': [
            [256, 128, 64],
            [512, 256, 128],
            [128, 64, 32]
        ],
        'dropout_rates': [0.2, 0.3, 0.4]
    }
    
    best_val_acc = 0.0
    best_params = {}
    results = []
    
    print("\n[DEBUG] Starting hyperparameter tuning...")
    
    for n_components in param_grid['n_pca_components']:
        for lr in param_grid['learning_rates']:
            for wd in param_grid['weight_decays']:
                for hidden_dims in param_grid['hidden_dims_configs']:
                    for dropout in param_grid['dropout_rates']:
                        print(f"\nTrying parameters:")
                        print(f"PCA components: {n_components}")
                        print(f"Learning rate: {lr}")
                        print(f"Weight decay: {wd}")
                        print(f"Hidden dims: {hidden_dims}")
                        print(f"Dropout: {dropout}")
                        
                        # Initialize model with current parameters
                        model = EnhancedDysarthriaClassifier(
                            input_dim=input_dim,
                            num_classes=num_classes,
                            hidden_dims=hidden_dims,
                            dropout_rate=dropout
                        ).to(device)
                        
                        # Train model
                        model, train_losses, val_losses = train_enhanced_model(
                            model, train_loader, val_loader, device,
                            num_epochs=30,  # Reduced epochs for faster tuning
                            lr=lr,
                            weight_decay=wd
                        )
                        
                        # Evaluate on validation set
                        model.eval()
                        correct = 0
                        total = 0
                        with torch.no_grad():
                            for inputs, labels in val_loader:
                                inputs, labels = inputs.to(device), labels.to(device)
                                outputs = model(inputs)
                                if isinstance(outputs, tuple):
                                    outputs = outputs[0]
                                _, predicted = outputs.max(1)
                                total += labels.size(0)
                                correct += predicted.eq(labels).sum().item()
                        
                        val_acc = 100. * correct / total
                        print(f"Validation Accuracy: {val_acc:.2f}%")
                        
                        results.append({
                            'n_components': n_components,
                            'lr': lr,
                            'weight_decay': wd,
                            'hidden_dims': hidden_dims,
                            'dropout': dropout,
                            'val_acc': val_acc
                        })
                        
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_params = {
                                'n_components': n_components,
                                'lr': lr,
                                'weight_decay': wd,
                                'hidden_dims': hidden_dims,
                                'dropout_rate': dropout
                            }
    
    print("\n[DEBUG] Hyperparameter tuning completed!")
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save results to a file
    import json
    with open('hyperparameter_tuning_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return best_params, best_val_acc


def main(feature_extractor_type: str = 'enhanced', **feature_extractor_kwargs):
    """
    Main pipeline function.
    
    Args:
        feature_extractor_type: Type of feature extractor to use ('enhanced' or 'pca')
        **feature_extractor_kwargs: Additional arguments for feature extractors
    """
    print(f"[DEBUG] Starting main pipeline using {feature_extractor_type} feature extractor")
    
    # Create necessary directories
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    processed_dir = 'processed_features'
    os.makedirs(processed_dir, exist_ok=True)

    # Initialize feature manager with explicit directory
    feature_manager = FeatureManager(feature_dir=processed_dir)

    # --- Data Loading ---
    data_dir = "C:\\Users\\Sowmya\\research\\processed_features\\data"  # Directory containing the audio files
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory '{data_dir}' not found. Please make sure it exists.")
        
    metadata_csv = "verified_metadata.csv"
    if not os.path.exists(metadata_csv):
        raise ValueError(f"Metadata file '{metadata_csv}' not found. Please run generate_labels.py first.")
        
    print(f"[DEBUG] Loading metadata from {metadata_csv}")
    metadata = pd.read_csv(metadata_csv)
    metadata['file_path'] = metadata['file_path'].apply(lambda x: os.path.join(data_dir, x))
    
    # Verify audio files exist
    existing_files = metadata['file_path'].apply(os.path.exists)
    if not all(existing_files):
        missing_files = metadata[~existing_files]['file_path'].tolist()
        print("[WARNING] Some audio files are missing:")
        for f in missing_files[:5]:  # Show first 5 missing files
            print(f"  - {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
        metadata = metadata[existing_files]
        print(f"[INFO] Proceeding with {len(metadata)} existing files")
    
    # Filter out 'control' samples
    severity_classes = ['Mild', 'Moderate', 'Severe', 'Very_Severe']
    metadata = metadata[metadata['severity'].isin(severity_classes)]
    label_map = {label: idx for idx, label in enumerate(severity_classes)}
    metadata['label'] = metadata['severity'].map(label_map)
    print('Label mapping:', label_map)
    print(f"[DEBUG] Total samples after filtering: {len(metadata)}")

    if len(metadata) == 0:
        raise ValueError("No valid samples found after filtering. Please check your data and metadata.")

    # --- Data Splitting ---
    audio_paths = metadata['file_path'].tolist()
    labels = metadata['label'].values
    
    train_paths, temp_paths, y_train, y_temp = train_test_split(
        audio_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_paths, test_paths, y_val, y_test = train_test_split(
        temp_paths, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # --- Feature Extraction ---
    # Process training split first to get fitted models
    print("[DEBUG] Processing training split...")
    features = {}
    features['train'] = extract_features(
        feature_extractor_type=feature_extractor_type,
        paths=train_paths,
        sr=16000,
        feature_manager=feature_manager,
        split='train',
        y=y_train,
        **feature_extractor_kwargs
    )

    # Now process validation and test splits using fitted models
    print("[DEBUG] Processing validation split...")
    features['val'] = extract_features(
        feature_extractor_type=feature_extractor_type,
        paths=val_paths,
        sr=16000,
        feature_manager=feature_manager,
        split='val',
        y=y_val,
        **feature_extractor_kwargs
    )

    print("[DEBUG] Processing test split...")
    features['test'] = extract_features(
        feature_extractor_type=feature_extractor_type,
        paths=test_paths,
        sr=16000,
        feature_manager=feature_manager,
        split='test',
        y=y_test,
        **feature_extractor_kwargs
    )

    # --- DataLoaders ---
    batch_size = 32
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(features['train']['features']), 
        torch.LongTensor(features['train']['labels'])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(features['val']['features']), 
        torch.LongTensor(features['val']['labels'])
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(features['test']['features']), 
        torch.LongTensor(features['test']['labels'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Model Training ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEBUG] Using device: {device}")
    
    # Initialize evaluator
    evaluator = BaselineModelEvaluator(save_dir=output_dir)
    all_metrics = {}

    # Train and evaluate baseline SVM model
    print("\n[DEBUG] Training SVM baseline model...")
    svm_model = MFCCSVMClassifier()
    svm_model.fit(train_paths, features['train']['labels'])
    svm_metrics = evaluator.evaluate_svm_model(
        model=svm_model,
        test_audio_paths=test_paths,
        test_labels=features['test']['labels']
    )
    all_metrics['SVM'] = svm_metrics
    print("\nSVM Model Metrics:")
    for metric_name, value in svm_metrics.items():
        if not isinstance(value, np.ndarray):  # Skip arrays like confusion matrix
            print(f"{metric_name}: {value}")

    # Train and evaluate baseline CNN model
    print("\n[DEBUG] Training CNN baseline model...")
    
    # Reshape the features for CNN input [batch_size, channels, features]
    train_features = features['train']['features'].reshape(len(features['train']['features']), 1, -1)
    val_features = features['val']['features'].reshape(len(features['val']['features']), 1, -1)
    test_features = features['test']['features'].reshape(len(features['test']['features']), 1, -1)
    
    # Create new dataloaders with reshaped features
    train_dataset_cnn = TensorDataset(
        torch.FloatTensor(train_features),
        torch.LongTensor(features['train']['labels'])
    )
    val_dataset_cnn = TensorDataset(
        torch.FloatTensor(val_features),
        torch.LongTensor(features['val']['labels'])
    )
    test_dataset_cnn = TensorDataset(
        torch.FloatTensor(test_features),
        torch.LongTensor(features['test']['labels'])
    )
    
    train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=batch_size, shuffle=True)
    val_loader_cnn = DataLoader(val_dataset_cnn, batch_size=batch_size, shuffle=False)
    test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=batch_size, shuffle=False)
    
    # Initialize CNN with correct input shape
    input_channels = 1
    feature_dim = features['train']['features'].shape[1]
    cnn_baseline = CNNClassifier(input_channels=input_channels, num_classes=len(severity_classes)).to(device)
    
    cnn_baseline, _, _ = train_enhanced_model(
        cnn_baseline, train_loader_cnn, val_loader_cnn, device,
        num_epochs=50, lr=0.001, weight_decay=1e-4
    )
    cnn_metrics = evaluator.evaluate_cnn_model(
        model=cnn_baseline,
        test_loader=test_loader_cnn,
        device=device,
        model_name='cnn_baseline'
    )
    all_metrics['CNN'] = cnn_metrics
    print("\nCNN Baseline Model Metrics:")
    for metric_name, value in cnn_metrics.items():
        if not isinstance(value, np.ndarray):
            print(f"{metric_name}: {value}")

    # Initialize and train enhanced model
    print("\n[DEBUG] Training enhanced model...")
    input_dim = features['train']['features'].shape[1]
    num_classes = len(severity_classes)
    
    if feature_extractor_type == 'pca':
        print("\n[DEBUG] Running hyperparameter tuning for PCA model...")
        best_params, best_val_acc = tune_hyperparameters(
            train_loader, val_loader, device, input_dim, num_classes
        )
        
        # Train final model with best parameters
        model = EnhancedDysarthriaClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=best_params['hidden_dims'],
            dropout_rate=best_params['dropout_rate']
        ).to(device)
        
        model, train_losses, val_losses = train_enhanced_model(
            model, train_loader, val_loader, device,
            num_epochs=50,  # Back to full training
            lr=best_params['lr'],
            weight_decay=best_params['weight_decay']
        )
    else:
        model = EnhancedDysarthriaClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
        model, train_losses, val_losses = train_enhanced_model(
            model, train_loader, val_loader, device, num_epochs=50, lr=0.001, weight_decay=1e-4
        )

    # Evaluate enhanced model
    print("\n[DEBUG] Evaluating enhanced model...")
    enhanced_metrics = evaluator.evaluate_cnn_model(
        model=model,
        test_loader=test_loader,
        device=device,
        model_name=feature_extractor_type
    )
    all_metrics[f'Enhanced ({feature_extractor_type})'] = enhanced_metrics
    
    print(f"\n{feature_extractor_type.capitalize()} Model Metrics:")
    for metric_name, value in enhanced_metrics.items():
        if not isinstance(value, np.ndarray):
            print(f"{metric_name}: {value}")
    
    # Compare all models
    print("\n[DEBUG] Comparing all models...")
    print("Models being compared:")
    print("1. SVM (MFCC features)")
    print("2. CNN (raw features)")
    print(f"3. Enhanced ({feature_extractor_type} features)")
    evaluator.compare_models(all_metrics)

    # Save enhanced model
    model_save_dir = os.path.join(output_dir, feature_extractor_type)
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_extractor_type': feature_extractor_type,
        'feature_extractor_kwargs': feature_extractor_kwargs,
        'label_map': label_map,
        'input_dim': input_dim,
        'num_classes': num_classes
    }, os.path.join(model_save_dir, "model.pt"))
    print(f"[DEBUG] Model saved to {os.path.join(model_save_dir, 'model.pt')}")

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(model_save_dir, 'training_curves.png'))
    plt.close()

    return model, all_metrics


if __name__ == "__main__":
    # Example usage:
    # For enhanced feature extractor:
    # main(feature_extractor_type='enhanced', n_mfcc=13, n_mels=40)
    
    # For PCA feature extractor:
    # main(feature_extractor_type='pca', n_pca_components=20)
    
    # Default: use enhanced feature extractor
    # main()
    main(feature_extractor_type='pca', n_pca_components=20)
