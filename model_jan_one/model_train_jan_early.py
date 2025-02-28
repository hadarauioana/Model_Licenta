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
from model_arch_jan import HeartRateDataset, TransformerModel
from datetime import datetime
import json
import pickle
from sklearn.metrics import mean_absolute_error, r2_score

def save_predictions_to_csv(y_true, y_pred, times, user_ids, filename='final_predictions_with_time.csv'):
    min_length = min(len(y_true), len(y_pred), len(times), len(user_ids))
    y_true, y_pred, times, user_ids = (
        y_true[:min_length], y_pred[:min_length], times[:min_length], user_ids[:min_length]
    )

    predictions_df = pd.DataFrame({
        "user_id": user_ids,
        "time": times,
        "real_value": y_true,
        "predicted_value": y_pred
    })

    output_dir = '/home/ioana/Desktop/Model_Licenta/output'
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, filename)
    predictions_df.to_csv(file_path,mode='w', index=False)
    print(f"Predictions saved to: {file_path}")

def save_min_max(min_val, max_val, filename="scaler_params.json"):
    with open(filename, "w") as f:
        json.dump({"min": min_val, "max": max_val}, f)

# Function to load min and max values
def load_min_max(filename="scaler_params.json"):
    with open(filename, "r") as f:
        params = json.load(f)
    return params["min"], params["max"]

# Function to denormalize values
def denormalize(values, min_val, max_val):
    return values * (max_val - min_val) + min_val


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
    """
    Train the model with early stopping and save denormalized predictions including timestamps.
    """
    model.to(device)
    train_losses, val_losses = [], []
    mse_scores,mae_scores, r2_scores = [], [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Load min and max for denormalization
    with open('/home/ioana/Desktop/Model_Licenta/data/scaler_min_max.pkl', 'rb') as f:
        min_value, max_value = pickle.load(f)
    print(f"Scaler params loaded: min={min_value}, max={max_value}")
    results = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        for x_values, x_time_numeric, x_features, user_id, y in train_loader:
            x_values, x_time_numeric, x_features, user_id, y = (
                x_values.to(device),
                x_time_numeric.to(device),
                x_features.to(device),
                user_id.to(device),
                y.to(device)
            )

            optimizer.zero_grad()
            predictions = model(x_values, x_time_numeric, x_features, user_id)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        y_true, y_pred, times, user_ids_list = [], [], [], []

        with torch.no_grad():
            for x_values, x_time_numeric, x_features, user_id, y in val_loader:
                x_values, x_time_numeric, x_features, user_id, y = (
                    x_values.to(device),
                    x_time_numeric.to(device),
                    x_features.to(device),
                    user_id.to(device),
                    y.to(device)
                )
                predictions = model(x_values, x_time_numeric, x_features, user_id)
                loss = criterion(predictions, y)
                total_val_loss += loss.item()

                y_true_denorm = (y.cpu().numpy().flatten() * (max_value - min_value)) + min_value
                y_pred_denorm = (predictions.cpu().numpy().flatten() * (max_value - min_value)) + min_value

                batch_size, pred_len = predictions.shape
                times_batch = x_time_numeric[:, -1].cpu().numpy()  # Last timestamp per sequence
                times_batch = np.repeat(times_batch, pred_len)
                times_batch = pd.to_datetime(times_batch, unit='s', origin='unix').astype(str).tolist()

                user_ids_batch = np.repeat(user_id.cpu().numpy().flatten(), pred_len)
                user_ids_list.extend(user_ids_batch.tolist())

                y_true.extend(y_true_denorm.tolist())
                y_pred.extend(y_pred_denorm.tolist())
                times.extend(times_batch)


                # Debug after each batch
                # print(f"Lengths after batch - user_ids: {len(user_ids_list)}, times: {len(times)}, y_true: {len(y_true)}, y_pred: {len(y_pred)}")

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Evaluation metrics
        mse_norm = mean_squared_error(y_true, y_pred)
        mae_norm = mean_absolute_error(y_true, y_pred)
        r2_norm = r2_score(y_true, y_pred)

        mae_denorm = mean_absolute_error(y_true_denorm, y_pred_denorm)
        mse_denorm = mean_squared_error(y_true_denorm, y_pred_denorm)
        r2_denorm = r2_score(y_true_denorm, y_pred_denorm)

        mae_scores.append(mae_denorm)
        mse_scores.append(mse_denorm)
        r2_scores.append(r2_denorm)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        column_names = ["Time", "Epoch", "Train Loss", "Validation Loss", "MAE", "MSE", "R²"]
        # Save results to CSV (append mode)


        results_file = "/home/ioana/Desktop/Model_Licenta/output/training_results.csv"
        if not os.path.exists(results_file):
            pd.DataFrame(columns=column_names).to_csv(results_file, index=False)

        new_row = pd.DataFrame([[timestamp, epoch + 1, avg_train_loss, avg_val_loss, mae_denorm, mse_denorm, r2_denorm]],
                               columns=column_names)
        new_row.to_csv(results_file, mode='a', header=False, index=False)

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | MAE denorm: {mae_denorm:.4f} | MSE denorm: {mse_denorm:.4f} | R² denorm: {r2_denorm:.4f}"
        )

        # Early stopping and saving best predictions
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved.")

            # Save predictions for the best model
            save_predictions_to_csv(y_true, y_pred, times, user_ids_list, filename='best_model_predictions_with_time.csv')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(" Early stopping triggered due to no improvement.")
                torch.save(model.state_dict(), "best_model.pth")
                print("Best model saved.")
                # Save final predictions before stopping
                save_predictions_to_csv(y_true, y_pred, times, user_ids_list, filename='final_predictions_with_time.csv')
                break


# Load dataset
file_path: str = "/home/ioana/Desktop/Preprocesare_Date_Licenta/process_fitbit/fitbit_heartrate_merged_minutes.csv"
data: pd.DataFrame = pd.read_csv(file_path)

data["Time"] = pd.to_datetime(data["Time"])

# Map user IDs to indices
user_ids = data["Id"].unique()
user_id_map: dict[int, int] = {user_id: idx for idx, user_id in enumerate(user_ids)}

min_value = data["Value"].min()
max_value = data["Value"].max()

with open('/home/ioana/Desktop/Model_Licenta/data/scaler_min_max.pkl', 'wb') as f:
    pickle.dump((min_value, max_value), f)

print(f"Scaler parameters saved: min={min_value}, max={max_value}")

# Define hyperparameters
input_len: int = 30
pred_len: int = 10
overlap: int = 20
batch_size: int = 64

# Create dataset and DataLoader
dataset: Dataset = HeartRateDataset(data, input_len, pred_len, overlap, user_id_map)
scaler: MinMaxScaler = MinMaxScaler()
scaler.fit(data["Value"].values.reshape(-1, 1))

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {device}")

# Split dataset using GroupKFold
group_kfold = GroupKFold(n_splits=5)
groups = [user_id for (_, _, _, user_id, _) in dataset.data]
train_idx, val_test_idx = next(group_kfold.split(dataset.data, groups=groups))
val_test_groups = [groups[i] for i in val_test_idx]
train_data = torch.utils.data.Subset(dataset, train_idx)

val_idx, test_idx = next(GroupKFold(n_splits=2).split(val_test_idx, groups=val_test_groups))
val_data = torch.utils.data.Subset(dataset, val_idx)
test_data = torch.utils.data.Subset(dataset, test_idx)

# Data loaders
train_loader: DataLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader: DataLoader = DataLoader(val_data, batch_size=batch_size)
test_loader: DataLoader = DataLoader(test_data, batch_size=batch_size)

# Initialize model
model: nn.Module = TransformerModel(
    input_dim=1,
    time_dim=2,
    d_model=64,
    nhead=4,
    num_layers=3,
    num_users=len(user_ids),
    embedding_dim=16,
    pred_len=pred_len,
    dropout=0.1
)

criterion: nn.Module = nn.MSELoss()
optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model with early stopping
train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, patience=10)

# Save trained model
model_path: str = "heart_rate_model_fitbit.pth"
torch.save(model.state_dict(), model_path)
print(f"Trained model has been saved to: {model_path}")