
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
from utils import *
from models import NeuralNetwork, LogisticRegressionModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

start_time = time.time()

n_samples = 100
n_features = 1000

np.random.seed(0)
torch.manual_seed(0)

X = np.random.choice([0, 1], size=(n_samples, n_features), p=[0.8, 0.2])
y = np.logical_or(X[:, 0], X[:, 1]).astype(int)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

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

for epoch in range(1000):
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


# Random Forest Model
random_forest = RandomForestClassifier(n_estimators=n_features)
random_forest.fit(X_train.numpy(), y_train.numpy())
y_pred_rf = random_forest.predict(X_val.numpy())
val_loss_values_rf = []
val_loss_rf = None

total_accuracy_rf = accuracy_score(y_val.numpy(), y_pred_rf)
# Calculate and print accuracy for each model
avg_accuracy_nn, std_error_nn = calculate_accuracy(neural_network, y_val.view(-1, 1), X_val)
avg_accuracy_regression, std_error_regression = calculate_accuracy(regression_model, y_val.view(-1, 1), X_val)
avg_accuracy_rf = total_accuracy_rf

print(f"Neural Network Validation Accuracy: {avg_accuracy_nn:.2f} +/- {std_error_nn:.2f}")
print(f"Logistic Regression Model Validation Accuracy: {avg_accuracy_regression:.2f} +/- {std_error_regression:.2f}")
print(f"Random Forest Model Validation Accuracy: {avg_accuracy_rf:.2f}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time:.2f} seconds")

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
plt.title('Validation Loss Curves (1000 Features)')
plt.show()

y_val = np.array(y_val)
y_pred_rf = np.array(y_pred_rf)

# Calculate the proportions for measured=0 and measured=1
predicted_0 = [np.mean((y_val == 0) & (y_pred_rf == 1)), np.mean((y_val == 1) & (y_pred_rf == 1))]
predicted_1 = [1-(np.mean((y_val == 0) & (y_pred_rf == 1))), 1-(np.mean((y_val == 1) & (y_pred_rf == 1)))]

# Create labels for the categories
categories = ['Measured=0', 'Measured=1']

plt.bar([0,1], predicted_0, color='firebrick', label='Predicted 0', width=0.05)
plt.bar([0,1], predicted_1, bottom=predicted_0, color='paleturquoise', label='Predicted 1', width=0.05)
plt.xlabel('Measured Value')
plt.ylabel('Proportion with Predicted=1')
plt.title('Proportion of Predicted=1 for Measured Values')
plt.ylim(0, 1)  # Set the y-axis limit to 0-1
plt.show() 