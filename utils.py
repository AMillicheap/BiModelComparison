#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:27:29 2023

@author: alexmillicheap
"""

import torch
from sklearn.metrics import accuracy_score

def calculate_accuracy(model, y, X):
    with torch.no_grad():
        y_pred = (model(X) > 0.5).numpy().astype(int)
        accuracy = accuracy_score(y, y_pred)
        
    return accuracy
