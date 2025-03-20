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
from datetime import datetime
import json
import pickle
from sklearn.metrics import mean_absolute_error, r2_score

from comp_mar.architecture import HeartRateDataset, TransformerModel1, TransformerModel2


# 1- model 1 Trabsformers
# 2- model 2 Transf
# 3- model 1 LSTM
# 4 -model 2 LSTM

# 1,3 - train test pmdata
# 2,4 - trin test fitbit


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


def train_model_with_early_stopping(no:int,
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
    mse_scores, mae_scores, r2_scores = [], [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    if no % 2 == 1:
        # Load min and max for denormalization
        with open('/home/ioana/Desktop/Model_Licenta/data/scaler_min_max_pmdata.pkl', 'rb') as f:
            min_value, max_value = pickle.load(f)
        print(f"Scaler params loaded: min={min_value}, max={max_value}")

    else :
        # Load min and max for denormalization
        with open('/home/ioana/Desktop/Model_Licenta/data/scaler_min_max_fitbit.pkl', 'rb') as f:
            min_value, max_value = pickle.load(f)
        print(f"Scaler params loaded: min={min_value}, max={max_value}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0

        for x_values, x_time_numeric, x_features, x_log1, x_gradients, user_id, y in train_loader:
            x_values, x_time_numeric, x_features, x_log1, x_gradients, user_id, y = (
                x_values.to(device),
                x_time_numeric.to(device),
                x_features.to(device),
                x_log1.to(device),
                x_gradients.to(device),
                user_id.to(device),
                y.to(device)
            )

            optimizer.zero_grad()
            predictions = model(x_values, x_time_numeric, x_features, x_log1, x_gradients, user_id)
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
            for x_values, x_time_numeric, x_features, x_log1, x_gradients, user_id, y in val_loader:
                x_values, x_time_numeric, x_features, x_log1, x_gradients, user_id, y = (
                    x_values.to(device),
                    x_time_numeric.to(device),
                    x_features.to(device),
                    x_log1.to(device),
                    x_gradients.to(device),
                    user_id.to(device),
                    y.to(device)
                )
                predictions = model(x_values, x_time_numeric, x_features, x_log1, x_gradients, user_id)
                loss = criterion(predictions, y)
                total_val_loss += loss.item()

                y_true_denorm = (y.cpu().numpy().flatten() * (max_value - min_value)) + min_value
                y_pred_denorm = (predictions.cpu().numpy().flatten() * (max_value - min_value)) + min_value

                times_batch = x_time_numeric[:, -1].cpu().numpy()
                times_batch = pd.to_datetime(times_batch, unit='s', origin='unix').astype(str).tolist()
                user_ids_batch = user_id.cpu().numpy().flatten().tolist()

                y_true.extend(y_true_denorm.tolist())
                y_pred.extend(y_pred_denorm.tolist())
                times.extend(times_batch)
                user_ids_list.extend(user_ids_batch)

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Evaluation metrics
        mae_denorm = mean_absolute_error(y_true, y_pred)
        mse_denorm = mean_squared_error(y_true, y_pred)
        r2_denorm = r2_score(y_true, y_pred)

        mae_scores.append(mae_denorm)
        mse_scores.append(mse_denorm)
        r2_scores.append(r2_denorm)

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | MAE: {mae_denorm:.4f} | MSE: {mse_denorm:.4f} | R²: {r2_denorm:.4f}"
        )

        # Early stopping and saving best predictions
        best_model_name = f"best_model_{no}.pth"
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_name)
            print("Best model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered due to no improvement.")
                torch.save(model.state_dict(), best_model_name)
                print("Best model saved.")
                break


def test_model(no:int,
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    user_id_map: dict
) -> None:
    """
    Evaluate the trained model on the test set and save results.
    """
    model.to(device)
    model.eval()

    if no % 2 == 1:
        with open('/home/ioana/Desktop/Model_Licenta/data/scaler_min_max_pmdata.pkl', 'rb') as f:
            min_value, max_value = pickle.load(f)
    else:
        with open('/home/ioana/Desktop/Model_Licenta/data/scaler_min_max_fitbit.pkl', 'rb') as f:
            min_value, max_value = pickle.load(f)


    y_true_test, y_pred_test, times_test, user_ids_list_test = [], [], [], []

    with torch.no_grad():
        for x_values, x_time_numeric, x_features, x_log1, x_gradients, user_id, y in test_loader:
            x_values, x_time_numeric, x_features, x_log1, x_gradients, user_id, y = (
                x_values.to(device),
                x_time_numeric.to(device),
                x_features.to(device),
                x_log1.to(device),
                x_gradients.to(device),
                user_id.to(device),
                y.to(device)
            )

            predictions = model(x_values, x_time_numeric, x_features, x_log1, x_gradients, user_id)

            y_true_denorm = (y.cpu().numpy().flatten() * (max_value - min_value)) + min_value
            y_pred_denorm = (predictions.cpu().numpy().flatten() * (max_value - min_value)) + min_value

            times_batch = x_time_numeric[:, -1].cpu().numpy()
            times_batch = pd.to_datetime(times_batch, unit='s', origin='unix').astype(str).tolist()

            user_ids_batch = user_id.cpu().numpy().flatten().tolist()
            reverse_user_id_map = {v: k for k, v in user_id_map.items()}
            user_id_list_expanded = [reverse_user_id_map[uid] for uid in user_ids_batch]

            y_true_test.extend(y_true_denorm.tolist())
            y_pred_test.extend(y_pred_denorm.tolist())
            times_test.extend(times_batch)
            user_ids_list_test.extend(user_id_list_expanded)

    # Compute evaluation metrics
    mae_test = mean_absolute_error(y_true_test, y_pred_test)
    mse_test = mean_squared_error(y_true_test, y_pred_test)
    r2_test = r2_score(y_true_test, y_pred_test)

    print(f"Test MODEL {no}")
    print(f"Test Set Metrics: MAE={mae_test:.4f}, MSE={mse_test:.4f}, R²={r2_test:.4f}")

    results_filec_csv = f"test_results_{no}.csv"

    results_file = f"/home/ioana/Desktop/Model_Licenta/output/{results_filec_csv}"  # Fixed closing brace

    # Create a new row for the results
    new_row = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mae_test, mse_test, r2_test]],
                           columns=["Time", "MAE", "MSE", "R²"])

    # Append to CSV, adding header only if the file does not exist
    new_row.to_csv(results_file, mode='a', header=not os.path.exists(results_file), index=False)

# ---------------------------------------------------------------------------------------------------------------------------- MAIN ---------------------------------------------
# # Load dataset
fitbit_path: str = "/home/ioana/Desktop/Preprocesare_Date_Licenta/process_fitbit/fitbit_heartrate_merged_minutes.csv"
pmdata_path : str ="/home/ioana/Desktop/Preprocesare_Date_Licenta/process_pmdata/filter_merged_processed_data_pmdata.csv"
#
# # ********************************************************************************************* BIG DATASET  - FIRST MODEL ***************************************************
# pmdata: pd.DataFrame = pd.read_csv(pmdata_path)
#
# pmdata["Time"] = pd.to_datetime(pmdata["Time"])
#
# # Map user IDs to indices
# user_ids = pmdata["Id"].unique()
# user_id_map: dict[int, int] = {user_id: idx for idx, user_id in enumerate(user_ids)}
#
# min_value = pmdata["Value"].min()
# max_value = pmdata["Value"].max()
#
# with open('/home/ioana/Desktop/Model_Licenta/data/scaler_min_max_pmdata.pkl', 'wb') as f:
#     pickle.dump((min_value, max_value), f)
#
# print(f"Scaler parameters saved  ----PMDATA---- : min={min_value}, max={max_value}")
#
#
# # Define hyperparameters
# input_len: int = 30
# pred_len: int = 10
# overlap: int = 20
# batch_size: int = 64
#
# # Create dataset and DataLoader
# dataset: Dataset = HeartRateDataset(pmdata, input_len, pred_len, overlap, user_id_map)
# scaler: MinMaxScaler = MinMaxScaler()
# scaler.fit(pmdata["Value"].values.reshape(-1, 1))
#
# device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device in use: {device}")
#
# # Split dataset using GroupKFold
# group_kfold = GroupKFold(n_splits=5)
# groups = [user_id for (_, _, _, _,_,user_id, _) in dataset.data]
# train_idx, val_test_idx = next(group_kfold.split(dataset.data, groups=groups))
# val_test_groups = [groups[i] for i in val_test_idx]
# train_data = torch.utils.data.Subset(dataset, train_idx)
#
# val_idx, test_idx = next(GroupKFold(n_splits=2).split(val_test_idx, groups=val_test_groups))
# val_data = torch.utils.data.Subset(dataset, val_idx)
# test_data = torch.utils.data.Subset(dataset, test_idx)
#
# # Data loaders
# train_loader: DataLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# val_loader: DataLoader = DataLoader(val_data, batch_size=batch_size)
# test_loader: DataLoader = DataLoader(test_data, batch_size=batch_size)
# # Data loaders
# train_loader: DataLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# val_loader: DataLoader = DataLoader(val_data, batch_size=batch_size)
# test_loader: DataLoader = DataLoader(test_data, batch_size=batch_size)
#
# print("* PMDATA DataLoader succsess")
#
# model1_transformer: nn.Module = TransformerModel1(
#     input_dim=1,
#     time_dim=2,
#     d_model=64,
#     nhead=4,
#     num_layers=3,
#     num_users=len(user_ids),
#     embedding_dim=16,
#     pred_len=pred_len,
#     dropout=0.1
# )
#
# criterion: nn.Module = nn.MSELoss()
# optimizer: torch.optim.Optimizer = torch.optim.Adam(model1_transformer.parameters(), lr=0.001)
#
# print("--------------------------------------------- TRAIN MODEL TRANSFORMER 1-----------------------------------------------------")
# # Train the model with early stopping
# train_model_with_early_stopping(1,model1_transformer, train_loader, val_loader, criterion, optimizer, device, epochs=100, patience=10)
# model1_transformer.load_state_dict(torch.load("best_model_1.pth", map_location=device))
#
# # Save trained model
# # model_path: str = "heart_rate_model_t1.pth"
# # torch.save(model1_transformer.state_dict(), model_path)
# # print(f"Trained model has been saved to: {model_path}")
#
# # Set model to evaluation mode
# model1_transformer.eval()
# print("Model loaded successfully.")
# print("--------------------------------------------- TEST MODEL TRANSFORMER 1-----------------------------------------------------")
# test_model(1,model1_transformer, test_loader, nn.MSELoss(), device, user_id_map)


# ********************************************************************************************* SMALL DATASET  - SECOND MODEL ***************************************************
fitbitdata: pd.DataFrame = pd.read_csv(fitbit_path)

fitbitdata["Time"] = pd.to_datetime(fitbitdata["Time"])

# Map user IDs to indices
user_ids_f= fitbitdata["Id"].unique()
user_id_map_f: dict[int, int] = {user_id_f: idx for idx, user_id_f in enumerate(user_ids_f)}

min_value_f = fitbitdata["Value"].min()
max_value_f = fitbitdata["Value"].max()

with open('/home/ioana/Desktop/Model_Licenta/data/scaler_min_max_fitbit.pkl', 'wb') as fg:
    pickle.dump((min_value_f, max_value_f), fg)

print(f"Scaler parameters saved  ----FITBI DATA---- : min={min_value_f}, max={max_value_f}")


# Define hyperparameters for second transformer model
input_len_st: int = 30
pred_len_st: int = 10
overlap_st: int = 20
batch_size_st: int = 64

# Create dataset and DataLoader
dataset_st: Dataset = HeartRateDataset(fitbitdata, input_len_st, pred_len_st, overlap_st, user_id_map_f)
scaler_st: MinMaxScaler = MinMaxScaler()
scaler_st.fit(fitbitdata["Value"].values.reshape(-1, 1))

device_st: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {device_st}")

# Split dataset using GroupKFold
group_kfold_st = GroupKFold(n_splits=5)
groups_st = [user_id for (_, _, _, _,_,user_id, _) in dataset_st.data]
train_idx_st, val_test_idx_st = next(group_kfold_st.split(dataset_st.data, groups=groups_st))
val_test_groups = [groups_st[i] for i in val_test_idx_st]
train_data = torch.utils.data.Subset(dataset_st, train_idx_st)

val_idx_st, test_idx_st = next(GroupKFold(n_splits=2).split(val_test_idx_st, groups=val_test_groups))
val_data_st = torch.utils.data.Subset(dataset_st, val_idx_st)
test_data_st = torch.utils.data.Subset(dataset_st, test_idx_st)

# Data loaders
train_loader_st: DataLoader = DataLoader(train_data, batch_size=batch_size_st, shuffle=True)
val_loader_st: DataLoader = DataLoader(val_data_st, batch_size=batch_size_st)
test_loader_st: DataLoader = DataLoader(test_data_st, batch_size=batch_size_st)
# Data loaders
train_loader_st: DataLoader = DataLoader(train_data, batch_size=batch_size_st, shuffle=True)
val_loader_st: DataLoader = DataLoader(val_data_st, batch_size=batch_size_st)
test_loader_st: DataLoader = DataLoader(test_data_st, batch_size=batch_size_st)

print("* PMDATA DataLoader succsess")

# Load Model & Transfer Learning
checkpoint = torch.load("/home/ioana/Desktop/Model_Licenta/comp_mar/best_model_1.pth")
model2_transformer: nn.Module = TransformerModel2(
    input_dim=1,
    time_dim=2,
    d_model=64,
    nhead=4,
    num_layers=3,
    num_users=len(user_ids_f),
    embedding_dim=16,
    pred_len=pred_len_st,
    dropout=0.1
)
model2_transformer.to(device_st)
model2_transformer.load_state_dict(checkpoint, strict=False)
#
# # Freeze layers except final output
# for param in model.parameters():
#     param.requires_grad = False
# model.fc_out = nn.Linear(64, new_pred_len).to(device)
# for param in model.fc_out.parameters():
#     param.requires_grad = True

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model2_transformer.parameters()), lr=0.001)

# Initialize Transformer model2
criterion_t2: nn.Module = nn.MSELoss()
optimizer_t2: torch.optim.Optimizer = torch.optim.Adam(model2_transformer.parameters(), lr=0.001)

# Train the model with early stopping
# train_model_with_early_stopping(2, model2_transformer, train_loader_st, val_loader_st, criterion_t2, optimizer_t2, device_st, epochs=100, patience=10)
model2_transformer.load_state_dict(torch.load("/home/ioana/Desktop/Model_Licenta/comp_mar/best_model_2.pth", map_location=device_st))

# Save trained model
# model_path: str = "heart_rate_model_fitbit.pth"
# torch.save(model2_transformer.state_dict(), model_path)
# print(f"Trained model has been saved to: {model_path}")

# Set model to evaluation mode
model2_transformer.eval()
print("Model loaded successfully.")
print("---------------------------------------------TEST TRANSFORMER 2-----------------------------------------------------")
test_model(2, model2_transformer, test_loader_st, nn.MSELoss(), device_st, user_id_map_f)