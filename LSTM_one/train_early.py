import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os
from lstm_bilstm_att import HeartRateDataset
from datetime import datetime
import json
import pickle
from sklearn.metrics import mean_absolute_error, r2_score



def train_model_with_early_stopping(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int,
        patience: int
) -> None:
    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for x_values, x_time_numeric, x_features, x_lag1, x_gradients, _, y in train_loader:
            x_values, x_time_numeric, x_features, x_lag1, x_gradients, y = (
                x_values.to(device),
                x_time_numeric.to(device),
                x_features.to(device),
                x_lag1.to(device),
                x_gradients.to(device),
                y.to(device)
            )
            optimizer.zero_grad()
            predictions, _ = model(x_values, x_time_numeric, x_features, x_lag1, x_gradients)
            loss = criterion(predictions.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for x_values, x_time_numeric, x_features, x_lag1, x_gradients, _, y in val_loader:
                x_values, x_time_numeric, x_features, x_lag1, x_gradients, y = (
                    x_values.to(device),
                    x_time_numeric.to(device),
                    x_features.to(device),
                    x_lag1.to(device),
                    x_gradients.to(device),
                    y.to(device)
                )
                predictions, _ = model(x_values, x_time_numeric, x_features, x_lag1, x_gradients)
                loss = criterion(predictions.squeeze(), y)
                total_val_loss += loss.item()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered due to no improvement.")
                break
