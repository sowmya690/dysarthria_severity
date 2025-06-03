import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Union
from models.baseline_models import MFCCSVMClassifier, CNNClassifier
from models.enhanced_model import EnhancedDysarthriaClassifier


class BaselineModelEvaluator:
    """Evaluator for comparing baseline and enhanced models."""
    
    def __init__(self, save_dir: str = "evaluation_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def evaluate_svm_model(self, 
                          model: MFCCSVMClassifier,
                          test_audio_paths: List[str],
                          test_labels: np.ndarray) -> Dict[str, Any]:
        """Evaluate MFCC+SVM model."""
        predictions = model.predict(test_audio_paths)
        probabilities = model.predict_proba(test_audio_paths)
        
        metrics = self._calculate_metrics(test_labels, predictions)
        metrics['probabilities'] = probabilities
        
        self._save_results(metrics, model_name="mfcc_svm")
        return metrics
    
    def evaluate_cnn_model(self,
                          model: Union[CNNClassifier, EnhancedDysarthriaClassifier],
                          test_loader: torch.utils.data.DataLoader,
                          device: torch.device,
                          model_name: str = "cnn") -> Dict[str, Any]:
        """Evaluate CNN or Enhanced model."""
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # Handle tuple output from enhanced model
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Get logits only
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        metrics = self._calculate_metrics(np.array(all_labels), np.array(all_preds))
        metrics['probabilities'] = np.array(all_probs)
        
        self._save_results(metrics, model_name=model_name)
        return metrics
    
    def _calculate_metrics(self, 
                          true_labels: np.ndarray,
                          predictions: np.ndarray) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        cm = confusion_matrix(true_labels, predictions)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "predictions": predictions,
            "true_labels": true_labels
        }
    
    def _save_results(self, metrics: Dict[str, Any], model_name: str) -> None:
        """Save evaluation results and plots."""
        model_dir = os.path.join(self.save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save metrics summary
        with open(os.path.join(model_dir, "metrics.txt"), "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1-score: {metrics['f1_score']:.4f}\n")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], 
                   annot=True, 
                   fmt="d", 
                   cmap='Blues',
                   xticklabels=['Mild', 'Moderate', 'Severe'],
                   yticklabels=['Mild', 'Moderate', 'Severe'])
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
        plt.close()
        
        # Save predictions and probabilities
        np.save(os.path.join(model_dir, "predictions.npy"), metrics['predictions'])
        np.save(os.path.join(model_dir, "true_labels.npy"), metrics['true_labels'])
        if 'probabilities' in metrics:
            np.save(os.path.join(model_dir, "probabilities.npy"), metrics['probabilities'])
    
    def compare_models(self, 
                      metrics_dict: Dict[str, Dict[str, Any]]) -> None:
        """Compare performance of different models."""
        comparison_dir = os.path.join(self.save_dir, "model_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Create comparison table
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(metrics_dict.keys())
        
        comparison_data = []
        for metric in metrics:
            row = [metric]
            for model_name in model_names:
                row.append(f"{metrics_dict[model_name][metric]:.4f}")
            comparison_data.append(row)
        
        # Save comparison table
        with open(os.path.join(comparison_dir, "model_comparison.txt"), "w") as f:
            f.write("Model Comparison\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Metric':<15}" + "".join(f"{name:<15}" for name in model_names) + "\n")
            f.write("-" * 50 + "\n")
            for row in comparison_data:
                f.write("".join(f"{item:<15}" for item in row) + "\n")
        
        # Plot comparison bar chart
        plt.figure(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            values = [metrics_dict[model_name][metric] for metric in metrics]
            plt.bar(x + i * width, values, width, label=model_name)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Comparison')
        plt.xticks(x + width * (len(model_names) - 1) / 2, metrics)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "model_comparison.png"))
        plt.close() 