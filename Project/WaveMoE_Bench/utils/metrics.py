import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
import os

def calculate_classification_metrics(y_true, y_pred, num_classes=3):
    """
    Calculate full classification metrics.
    Returns: dict with accuracy, precision, recall, f1
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='macro', zero_division=0)
    }
    return metrics


def count_model_parameters(model, trainable_only=True):
    """
    Count total parameters of the model.
    Args:
        model: PyTorch model instance
        trainable_only: Only count parameters with requires_grad=True
    Returns:
        total_params: Total parameter count
        total_m: Total parameters in millions (M)
    """
    if trainable_only:
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = list(model.parameters())

    total_params = sum(p.numel() for p in params)
    total_m = total_params / 1e6
    return total_params, round(total_m, 4)

def save_training_log(epoch, train_loss, train_metrics, val_loss, val_metrics,
                      save_path="./output_figures/training_log.csv"):
    """
    Append training metrics to CSV log file.
    Automatically creates header on first run.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file_exists = os.path.isfile(save_path)

    with open(save_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "epoch", "train_loss", "train_acc", "train_f1",
                "val_loss", "val_acc", "val_f1"
            ])
        writer.writerow([
            epoch,
            round(train_loss, 6),
            round(train_metrics["accuracy"], 6),
            round(train_metrics["f1"], 6),
            round(val_loss, 6),
            round(val_metrics["accuracy"], 6),
            round(val_metrics["f1"], 6)
        ])
