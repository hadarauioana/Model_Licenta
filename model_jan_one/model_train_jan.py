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

def save_predictions_to_csv(y_true, y_pred, times, user_ids, filename):
    min_length = min(len(y_true), len(y_pred), len(times), len(user_ids))
    print("min length save_pred ",min_length)
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
    predictions_df.to_csv(file_path, index=False,mode='w')
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

def train_model(
    user_id_map:dict,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int) -> None:
    """
    Train the model and save denormalized predictions including timestamps.
    """
    model.to(device)
    train_losses, val_losses = [], []
    mse_scores, mae_scores, r2_scores = [], [], []
    best_val_loss = float('inf')

    # Load min and max for denormalization
    with open('/home/ioana/Desktop/Model_Licenta/data/scaler_min_max.pkl', 'rb') as f:
        min_value, max_value = pickle.load(f)

    print(f"Scaler params loaded: min={min_value}, max={max_value}")

    # Lists to store metrics
    results = []
    import time
    start_total_time = time.time()  # Start overall time tracking

    for epoch in range(epochs):
        start_epoch_time = time.time()  # Track epoch time
        # Training phase
        model.train()
        total_train_loss = 0.0
        for x_values, x_time_numeric, x_features,x_log1, x_log5, user_id, y in train_loader:
            x_values, x_time_numeric, x_features,x_log1, x_log5, user_id, y = (
                x_values.to(device),
                x_time_numeric.to(device),
                x_features.to(device),
                x_log1.to(device),
                x_log5.to(device),
                user_id.to(device),
                y.to(device)
            )

            optimizer.zero_grad()
            predictions = model(x_values, x_time_numeric, x_features,x_log1, x_log5, user_id)
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
            for x_values, x_time_numeric, x_features,x_log1, x_log5, user_id, y in val_loader:
                x_values, x_time_numeric, x_features,x_log1, x_log5, user_id, y = (
                    x_values.to(device),
                    x_time_numeric.to(device),
                    x_features.to(device),
                    x_log1.to(device),
                    x_log5.to(device),
                    user_id.to(device),
                    y.to(device)
                )
                predictions = model(x_values, x_time_numeric, x_features,x_log1, x_log5, user_id)
                loss = criterion(predictions, y)
                total_val_loss += loss.item()

                y_true_denorm = (y.cpu().numpy().flatten() * (max_value - min_value)) + min_value
                y_pred_denorm = (predictions.cpu().numpy().flatten() * (max_value - min_value)) + min_value

                batch_size, pred_len = predictions.shape
                times_batch = x_time_numeric[:, -1].cpu().numpy()  # Last timestamp per sequence
                # Generate incremental timestamps
                expanded_times = []
                for time in times_batch:
                    expanded_times.extend([pd.to_datetime(time, unit='s', origin='unix') + pd.Timedelta(minutes=i)
                                           for i in range(1, pred_len + 1)])  # Add 1 to pred_len minutes incrementally

                # Convert timestamps to string format
                expanded_times = [t.strftime('%Y-%m-%d %H:%M:%S') for t in expanded_times]

                # Expand user IDs for each predicted step
                user_ids_batch = np.repeat(user_id.cpu().numpy().flatten(), pred_len)

                # Append to lists
                user_ids_list.extend(user_ids_batch.tolist())
                # print(user_id_map)
                # for user_id in user_ids_list:
                #     matching_key = next((k for k, v in user_id_map.items() if v == user_id), None)
                    # print(matching_key)
                reverse_user_id_map = {v: k for k, v in user_id_map.items()}
                user_dList_duplicate = [reverse_user_id_map[value] for value in user_ids_list]
                y_true.extend(y_true_denorm.tolist())
                y_pred.extend(y_pred_denorm.tolist())
                times.extend(expanded_times)  # Use incremental timestamps instead of repeated ones

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

        import time
        end_epoch_time = time.time()
        epoch_duration = end_epoch_time - start_epoch_time  # Time for one epoch

        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Epoch Duration: {epoch_duration:.2f} sec "
            f"Val Loss: {avg_val_loss:.4f} | MAE denorm: {mae_denorm:.4f} | MSE denorm: {mse_denorm:.4f} | R² denorm: {r2_denorm:.4f}"
        )

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved.")

            # Save predictions for the best model
            save_predictions_to_csv(y_true, y_pred, times, user_dList_duplicate, filename='best_model_predictions_with_time.csv')

    total_training_time = time.time() - start_total_time
    print(f"Total training time: {total_training_time:.2f} sec")

   # Save final model after all epochs
    torch.save(model.state_dict(), "final_model.pth")
    print("Final model saved.")

    # Save final predictions
    save_predictions_to_csv(y_true, y_pred, times, user_dList_duplicate, filename='final_predictions_with_time.csv')


   # TEST
    y_true_test, y_pred_test, times_test, user_ids_list_test = [], [], [], []

    with torch.no_grad():
        for x_values_test, x_time_numeric_test, x_features_test, x_log1_test, x_log5_test,user_id_test, y_test in test_loader:
            x_values_test, x_time_numeric_test, x_features_test,x_log1_test, x_log5_test, user_id_test, y_test = (
                x_values_test.to(device),
                x_time_numeric_test.to(device),
                x_features_test.to(device),
                x_log1_test.to(device),
                x_log5_test.to(device),
                user_id_test.to(device),
                y_test.to(device)
            )

            predictions_test = model(x_values_test, x_time_numeric_test, x_features_test, x_log1_test, x_log5_test, user_id_test)

            # Denormalize predictions and ground truth
            y_true_denorm_test = (y_test.cpu().numpy().flatten() * (max_value - min_value)) + min_value
            y_pred_denorm_test = (predictions_test.cpu().numpy().flatten() * (max_value - min_value)) + min_value

            # Extract timestamps
            times_batch_test = x_time_numeric_test[:, -1].cpu().numpy()
            expanded_times_test = []
            pred_len_test = predictions_test.shape[1]  # Number of prediction steps

            for time in times_batch_test:
                expanded_times_test.extend([pd.to_datetime(time, unit='s', origin='unix') + pd.Timedelta(minutes=i)
                                       for i in range(1, pred_len + 1)])

            expanded_times_test = [t.strftime('%Y-%m-%d %H:%M:%S') for t in expanded_times_test]

            # Expand user IDs for each prediction step
            user_ids_batch = np.repeat(user_id_test.cpu().numpy().flatten(), pred_len_test)

            # Map numerical user ID back to original user ID
            reverse_user_id_map = {v: k for k, v in user_id_map.items()}
            user_id_list_expanded_test = [reverse_user_id_map[uid] for uid in user_ids_batch]

            # Append results
            y_true_test.extend(y_true_denorm_test.tolist())
            y_pred_test.extend(y_pred_denorm_test.tolist())
            times_test.extend(expanded_times_test)
            user_ids_list_test.extend(user_id_list_expanded_test)

    # Compute evaluation metrics
    mae_test = mean_absolute_error(y_true_test, y_pred_test)
    mse_test = mean_squared_error(y_true_test, y_pred_test)
    r2_test= r2_score(y_true_test, y_pred_test)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    column_names_test = ["Time", "MAE", "MSE", "R²"]

    # Save results to CSV (append mode)
    results_file = "/home/ioana/Desktop/Model_Licenta/output/test_results.csv"
    if not os.path.exists(results_file):
        pd.DataFrame(columns=column_names_test).to_csv(results_file, index=False)

    new_row = pd.DataFrame([[timestamp, mae_test, mse_test, r2_test]],
                           columns=column_names_test)
    new_row.to_csv(results_file, mode='a', header=False, index=False)

    print(f"Test Set Evaluation Metrics:")
    print(f"  MAE: {mae_test:.4f}")
    print(f"  MSE: {mse_test:.4f}")
    print(f"  R² Score: {r2_test:.4f}")

    # Save predictions to CSV
    # print(y_true_test[0])
    # print(y_pred_test[0])
    # print(times_test[0])
    # print(times_test)
    # print(user_ids_list_test[0])
    # print(user_ids_list_test)
    save_predictions_to_csv(y_true_test, y_pred_test, times_test, user_ids_list_test, filename="test_set_predictions.csv")



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
    print(f"Saved: min={min_value}, max={max_value}")

print(f"Scaler parameters saved: min={min_value}, max={max_value}")

# Define hyperparameters
input_len: int = 30
pred_len: int = 10
overlap: int = 20
batch_size: int = 64

# Create dataset and DataLoader
# print(data.head(50))
dataset: Dataset = HeartRateDataset(data, input_len, pred_len, overlap, user_id_map)
# x_values, x_time_numeric, x_features, user_id, y = dataset[0]
# y_true_denorm = (y.cpu().numpy().flatten() * (max_value - min_value)) + min_value
#
# x_true_denorm = (x_values.cpu().numpy().flatten() * (max_value - min_value)) + min_value
# print(x_true_denorm)
# print(y_true_denorm)
#
# times_batch = x_time_numeric[0].cpu().numpy()
# print(times_batch)
# print(f"Raw Timestamp: {times_batch}")
# timestamp = datetime.utcfromtimestamp(times_batch.item())  # Use .item() to extract scalar
# print(f"Converted UTC Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
#
# # Extract the user ID correctly
# user_ids_batch = user_id.item() if isinstance(user_id, torch.Tensor) else user_id  # NEW
# print(f"User ID: {user_ids_batch}")  # Should now print an integer
# print(f"User ID: {user_id_map.items()}")  # Should now print an integer

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {device}")



# Split dataset using GroupKFold
group_kfold = GroupKFold(n_splits=5)
groups = [user_id for (_, _, _, _,_,user_id, _) in dataset.data]
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
    d_model=128,
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
train_model(user_id_map,model, train_loader, val_loader, criterion, optimizer, device, epochs=100)

# Save trained model
model_path: str = "heart_rate_model_fitbit.pth"
torch.save(model.state_dict(), model_path)
print(f"Trained model has been saved to: {model_path}")