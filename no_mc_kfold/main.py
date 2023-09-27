#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 08:58:21 2023

@author: alexmillicheap
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from utils import *
from models import NeuralNetwork, LogisticRegressionModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

start_time = time.time()

n_samples = 100
n_features = 100
n_mc_samples = 5
np.random.seed(0)
torch.manual_seed(0)

X = np.random.choice([0, 1], size=(n_samples, n_features), p=[0.8, 0.2])
y = np.logical_or(X[:, 0], X[:, 1]).astype(int)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
y = y.view(-1)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

criterion = nn.BCELoss()

neural_network = NeuralNetwork(n_features)
optimizer_nn = optim.Adam(neural_network.parameters(), lr=0.001)
train_loss_values_nn = []
val_loss_values_nn = []

for epoch in range(1000):
    optimizer_nn.zero_grad()
    outputs_nn = neural_network(X_train)
    loss_nn = criterion(outputs_nn, y_train.view(-1, 1))
    loss_nn.backward()
    optimizer_nn.step()
    train_loss_values_nn.append(loss_nn.item())
    with torch.no_grad():
        outputs_val_nn = neural_network(X_val)
        val_loss_nn = criterion(outputs_val_nn, y_val.view(-1, 1))
        val_loss_values_nn.append(val_loss_nn.item())
neural_network.eval()
    
# Logistic Regression Model
regression_model = LogisticRegressionModel(n_features)
optimizer_regression = optim.Adam(regression_model.parameters(), lr=0.001)
train_loss_values_regression = []
val_loss_values_regression = []

for epoch in range(10000):
    optimizer_regression.zero_grad()
    outputs_regression = regression_model(X_train)
    loss_regression = criterion(outputs_regression, y_train.view(-1, 1))
    loss_regression.backward()
    optimizer_regression.step()
    train_loss_values_regression.append(loss_regression.item())
    with torch.no_grad():
        outputs_val_regression = regression_model(X_val)
        val_loss_regression = criterion(outputs_val_regression, y_val.view(-1, 1))
        val_loss_values_regression.append(val_loss_regression.item())
regression_model.eval()

predictions_rf = []
# Random Forest Model
for epoch in range(n_mc_samples):
    random_forest = RandomForestClassifier(n_estimators=n_features)
    random_forest.fit(X_train.numpy(), y_train.numpy())
    y_pred_rf = random_forest.predict(X_val.numpy())
    predictions_rf.append(y_pred_rf)
    
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time:.2f} seconds")

mean_pred_rf = np.mean(predictions_rf, axis=0)
prediction_interval_rf = np.percentile(predictions_rf, [2.5,97.5], axis=0)

threshold = 0.5
binary_predictions = (mean_pred_rf >= threshold).astype(int)

confusion_matrix_rf = confusion_matrix(y_val.numpy(), binary_predictions)
accuracy_rf = accuracy_score(y_val.numpy(), binary_predictions)
precision_rf = precision_score(y_val.numpy(), binary_predictions)
recall_rf = recall_score(y_val.numpy(), binary_predictions)
f1_score_rf = f1_score(y_val.numpy(), binary_predictions)
total_accuracy_rf = accuracy_score(y_val.numpy(), binary_predictions)
accuracy_values_rf = [accuracy_score(y_val.numpy(), (y_pred >= threshold).astype(int)) for y_pred in predictions_rf]
avg_accuracy_rf = np.mean(accuracy_values_rf)
std_error_rf = np.std(accuracy_values_rf)

val_loss_values_rf = []
val_loss_rf = None

avg_accuracy_nn, std_error_nn = calculate_accuracy(neural_network, y_val.view(-1, 1), X_val)
y_pred_nn = (neural_network(X_val) >= threshold).float()
precision_nn = precision_score(y_val.numpy(), y_pred_nn.numpy())
recall_nn = recall_score(y_val.numpy(), y_pred_nn.numpy())
f1_score_nn = f1_score(y_val.numpy(), y_pred_nn.numpy())
confusion_matrix_nn = confusion_matrix(y_val.numpy(), y_pred_nn.numpy())

avg_accuracy_regression, std_error_regression = calculate_accuracy(regression_model, y_val.view(-1, 1), X_val)
y_pred_regression = (regression_model(X_val) >= threshold).float()
precision_regression = precision_score(y_val.numpy(), y_pred_regression.numpy())
recall_regression = recall_score(y_val.numpy(), y_pred_regression.numpy())
f1_score_regression = f1_score(y_val.numpy(), y_pred_regression.numpy())
confusion_matrix_lr = confusion_matrix(y_val.numpy(), y_pred_regression.numpy())

avg_accuracy_nn, std_error_nn = calculate_accuracy(neural_network, y_val.view(-1, 1), X_val)
avg_accuracy_regression, std_error_regression = calculate_accuracy(regression_model, y_val.view(-1, 1), X_val)
avg_accuracy_rf = total_accuracy_rf

print(f"Neural Network Validation Accuracy: {avg_accuracy_nn:.2f} +/- {std_error_nn:.2f}")
print(f"Neural Network Precision: {precision_nn:.2f}")
print(f"Neural Network Recall: {recall_nn:.2f}")
print(f"Neural Network F1-Score: {f1_score_nn:.2f}")
print("Neural Network Confusion Matrix:")
print(confusion_matrix_nn)

print(f"Logistic Regression Model Validation Accuracy: {avg_accuracy_regression:.2f} +/- {std_error_regression:.2f}")
print(f"Logistic Regression Model Precision: {precision_regression:.2f}")
print(f"Logistic Regression Model Recall: {recall_regression:.2f}")
print(f"Logistic Regression Model F1-Score: {f1_score_regression:.2f}")
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix_lr)

print(f"Random Forest Model Validation Accuracy (Monte Carlo): {avg_accuracy_rf:.2f} +/- {std_error_rf:.2f}")
print(f"Random Forest Precision: {precision_rf:.2f}")
print(f"Random Forest Recall: {recall_rf:.2f}")
print(f"Random Forest F1-Score: {f1_score_rf:.2f}")
print("Random Forest Confusion Matrix:")
print(confusion_matrix_rf)

# Plot validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_loss_values_nn, label='Neural Network Train Loss')
plt.plot(val_loss_values_nn, label='Neural Network Validation Loss')
plt.plot(train_loss_values_regression, label='Logistic Regression Train Loss')
plt.plot(val_loss_values_regression, label='Logistic Regression Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.legend()
plt.title('Validation Loss Curves (100 Features)')
plt.show()

print("Random Forest Confusion Matrix:")
print(confusion_matrix_rf)

disp1 = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_rf)
disp1.plot(cmap='Reds')
plt.title('Confusion Matrix (Random Forest, 100 Features)') 
plt.show()

disp2 = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_nn)
disp2.plot(cmap='Greens')
plt.title('Confusion Matrix (Neural Network, 100 Features)') 
plt.show()

disp3 = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_lr)
disp3.plot(cmap='Blues')
plt.title('Confusion Matrix (Logistic Regression, 100 Features)') 
plt.show()
