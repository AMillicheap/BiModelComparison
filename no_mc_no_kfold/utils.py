
import numpy as np
import torch
from sklearn.metrics import accuracy_score

def calculate_accuracy(model, y_true, X):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        y_pred = (y_pred >= 0.5).float()  # Convert to binary predictions
        correct = (y_pred == y_true).sum().item()
        total = len(y_true)
        accuracy = correct / total
        # Calculate standard error
        std_error = np.sqrt(accuracy * (1 - accuracy) / total)
    return accuracy, std_error
