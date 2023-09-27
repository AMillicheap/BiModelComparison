# -*- coding: utf-8 -*-
"""
BMH-22814 Research Assistant Interview Task - main
Alex Millicheap
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

from utils import calculate_accuracy
from models import NeuralNetwork, LogisticRegressionModel

start_time = time.time()

np.random.seed(0)
torch.manual_seed(0)

n_samples = 100
n_features = 1000
n_splits = 5
n_bootstraps = 10

X = np.random.choice([0, 1], size=(n_samples, n_features), p=[0.8, 0.2])
y = np.logical_or(X[:, 0], X[:, 1]).astype(int)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

criterion = nn.BCELoss()

accuracy_values_nn = []
accuracy_values_regression = []
accuracy_values_rf = []

confusion_matrices_nn = []
confusion_matrices_regression = []
confusion_matrices_rf = []
print(n_features)
for _ in range(n_bootstraps):
    kfolds = KFold(n_splits=n_splits, shuffle=True, random_state=np.random.randint(1, 10000))
    total_accuracy_nn = 0
    total_accuracy_regression = 0
    total_accuracy_rf = 0
    confusion_matrix_nn = np.zeros((2, 2), dtype=int)
    confusion_matrix_regression = np.zeros((2, 2), dtype=int)
    confusion_matrix_rf = np.zeros((2, 2), dtype=int)
    
    print(_)
    
    for train_i, val_i in kfolds.split(X):
        X_train, X_val = X[train_i], X[val_i]
        y_train, y_val = y[train_i], y[val_i]
        
        # Neural Network
        neural_network = NeuralNetwork(n_features)
        optimizer_nn = optim.Adam(neural_network.parameters(), lr=0.0007)
        
        for epoch in range(1000):
            optimizer_nn.zero_grad()
            outputs_nn = neural_network(X_train)
            loss_nn = criterion(outputs_nn, y_train.view(-1, 1))
            loss_nn.backward()
            optimizer_nn.step()
        
        neural_network.eval()
        total_accuracy_nn += calculate_accuracy(neural_network, y_val.view(-1, 1), X_val)
        y_pred_nn = neural_network(X_val)
        y_pred_nn = (y_pred_nn >= 0.5).float()
        confusion_matrix_nn += confusion_matrix(y_val, y_pred_nn)
        
        # Logistic Regression Model
        regression_model = LogisticRegressionModel(n_features)
        optimizer_regression = optim.Adam(regression_model.parameters(), lr=0.01)
        
        for epoch in range(1000):
            optimizer_regression.zero_grad()
            outputs_regression = regression_model(X_train)
            loss_regression = criterion(outputs_regression, y_train.view(-1, 1))
            loss_regression.backward()
            optimizer_regression.step()
        
        regression_model.eval()
        total_accuracy_regression += calculate_accuracy(regression_model, y_val.view(-1, 1), X_val)
        y_pred_regression = regression_model(X_val)
        y_pred_regression = (y_pred_regression >= 0.5).float()
        confusion_matrix_regression += confusion_matrix(y_val, y_pred_regression)
        
        # Random Forest Model
        random_forest = RandomForestClassifier(n_estimators=n_features)
        random_forest.fit(X_train.numpy(), y_train.numpy())
        y_pred_rf = random_forest.predict(X_val.numpy())
        total_accuracy_rf += accuracy_score(y_val.numpy(), y_pred_rf)
        confusion_matrix_rf += confusion_matrix(y_val, y_pred_rf)
    
    accuracy_values_nn.append(total_accuracy_nn / n_splits)
    accuracy_values_regression.append(total_accuracy_regression / n_splits)
    accuracy_values_rf.append(total_accuracy_rf / n_splits)
    
    confusion_matrices_nn.append(confusion_matrix_nn)
    confusion_matrices_regression.append(confusion_matrix_regression)
    confusion_matrices_rf.append(confusion_matrix_rf)

avg_accuracy_nn = np.mean(accuracy_values_nn)
avg_accuracy_regression = np.mean(accuracy_values_regression)
avg_accuracy_rf = np.mean(accuracy_values_rf)

std_error_nn = np.std(accuracy_values_nn)
std_error_regression = np.std(accuracy_values_regression)
std_error_rf = np.std(accuracy_values_rf)

print(f"Neural Network Validation Accuracy: {avg_accuracy_nn:.2f} +/- {std_error_nn:.2f}")
print(f"Logistic Regression Model Validation Accuracy: {avg_accuracy_regression:.2f} +/- {std_error_regression:.2f}")
print(f"Random Forest Model Validation Accuracy: {avg_accuracy_rf:.2f} +/- {std_error_rf:.2f}")

# Calculate Precision, Recall, and F1-score for all models
precision_nn = np.zeros(n_bootstraps)
recall_nn = np.zeros(n_bootstraps)
f1_score_nn = np.zeros(n_bootstraps)

precision_regression = np.zeros(n_bootstraps)
recall_regression = np.zeros(n_bootstraps)
f1_score_regression = np.zeros(n_bootstraps)

precision_rf = np.zeros(n_bootstraps)
recall_rf = np.zeros(n_bootstraps)
f1_score_rf = np.zeros(n_bootstraps)

for i in range(n_bootstraps):
    tn, fp, fn, tp = confusion_matrices_nn[i].ravel()
    precision_nn[i] = tp / (tp + fp)
    recall_nn[i] = tp / (tp + fn)
    f1_score_nn[i] = 2 * (precision_nn[i] * recall_nn[i]) / (precision_nn[i] + recall_nn[i])

    tn, fp, fn, tp = confusion_matrices_regression[i].ravel()
    precision_regression[i] = tp / (tp + fp)
    recall_regression[i] = tp / (tp + fn)
    f1_score_regression[i] = 2 * (precision_regression[i] * recall_regression[i]) / (precision_regression[i] + recall_regression[i])

    tn, fp, fn, tp = confusion_matrices_rf[i].ravel()
    precision_rf[i] = tp / (tp + fp)
    recall_rf[i] = tp / (tp + fn)
    f1_score_rf[i] = 2 * (precision_rf[i] * recall_rf[i]) / (precision_rf[i] + recall_rf[i])

# Print Precision, Recall, and F1-score for all models
print(f"Neural Network Precision: {np.mean(precision_nn):.2f} +/- {np.std(precision_nn):.2f}")
print(f"Neural Network Recall: {np.mean(recall_nn):.2f} +/- {np.std(recall_nn):.2f}")
print(f"Neural Network F1-Score: {np.mean(f1_score_nn):.2f} +/- {np.std(f1_score_nn):.2f}")

print(f"Logistic Regression Model Precision: {np.mean(precision_regression):.2f} +/- {np.std(precision_regression):.2f}")
print(f"Logistic Regression Model Recall: {np.mean(recall_regression):.2f} +/- {np.std(recall_regression):.2f}")
print(f"Logistic Regression Model F1-Score: {np.mean(f1_score_regression):.2f} +/- {np.std(f1_score_regression):.2f}")

print(f"Random Forest Model Precision: {np.mean(precision_rf):.2f} +/- {np.std(precision_rf):.2f}")
print(f"Random Forest Model Recall: {np.mean(recall_rf):.2f} +/- {np.std(recall_rf):.2f}")
print(f"Random Forest Model F1-Score: {np.mean(f1_score_rf):.2f} +/- {np.std(f1_score_rf):.2f}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time:.2f} seconds")

disp1 = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_rf)
disp1.plot(cmap='Reds')
plt.title('Confusion Matrix (Random Forest, 1000 Features)') 
plt.show()

disp2 = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_nn)
disp2.plot(cmap='Greens')
plt.title('Confusion Matrix (Neural Network, 1000 Features)') 
plt.show()

disp3 = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_regression)
disp3.plot(cmap='Blues')
plt.title('Confusion Matrix (Logistic Regression, 1000 Features)') 
plt.show()
