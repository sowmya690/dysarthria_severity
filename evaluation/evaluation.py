import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self):
        print("[DEBUG] ModelEvaluator initialized")

    def evaluate_model(self, model, data_loader, device):
        print("[DEBUG] Starting evaluation")
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        cm = confusion_matrix(all_labels, all_preds)

        print(f"[DEBUG] Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "all_preds": all_preds,
            "all_labels": all_labels,
        }
        return metrics

    def generate_evaluation_report(self, metrics, save_dir="results"):
        os.makedirs(save_dir, exist_ok=True)
        print(f"[DEBUG] Saving evaluation report to {save_dir}")

        # Save metrics summary
        with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1-score: {metrics['f1_score']:.4f}\n")

        # Plot confusion matrix
        plt.figure(figsize=(6,5))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt="d", cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
        plt.close()
        print("[DEBUG] Evaluation report generated.")
