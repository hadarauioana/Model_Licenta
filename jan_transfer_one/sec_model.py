from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
import pandas as pd
import numpy as np
import torch
import math
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from typing import List, Tuple, Dict


# Dataset Class with Shape Details and Strong Typing
class HeartRateDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_len: int, pred_len: int, overlap: int, user_id_map: Dict[int, int]):
        """
        Args:
            df: DataFrame with columns ['Id', 'Time', 'Value']
            input_len: Length of input sequences
            pred_len: Length of output sequences
            overlap: Overlap between sliding windows
            user_id_map: Mapping of user IDs to integer indices
        """
        self.data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]] = []
        self.input_len = input_len
        self.pred_len = pred_len
        self.overlap = overlap
        self.user_id_map = user_id_map

        # Normalize values per user using MinMaxScaler
        scaler = MinMaxScaler()
        for patient_id, group in df.groupby("Id"):
            group = group.sort_values("Time").reset_index(drop=True)
            values = scaler.fit_transform(group["Value"].values.reshape(-1, 1)).flatten()  # Shape: [num_samples]

            # Temporal features: Shape: [num_samples, 2]
            group["Hour"] = group["Time"].dt.hour / 23.0
            group["DayOfWeek"] = group["Time"].dt.dayofweek / 6.0
            time_features = group[["Hour", "DayOfWeek"]].values

            # Sliding window extraction
            step = input_len - overlap
            for i in range(0, len(values) - input_len - pred_len + 1, step):
                x_values = values[i:i + input_len]                      # Shape: [input_len]
                x_time = group["Time"].values[i:i + input_len]          # Shape: [input_len]
                x_features = time_features[i:i + input_len]            # Shape: [input_len, 2]
                y = values[i + input_len:i + input_len + pred_len]     # Shape: [pred_len]
                user_id = self.user_id_map[patient_id]
                self.data.append((x_values, x_time, x_features, user_id, y))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_values: Shape [input_len]
            x_time_numeric: Shape [input_len]
            x_features: Shape [input_len, 2]
            user_id: Scalar
            y: Shape [pred_len]
        """
        x_values, x_time, x_features, user_id, y = self.data[idx]

        # Convert datetime to seconds since epoch - Shape: [input_len]
        x_time_numeric = pd.to_datetime(x_time).astype(np.int64) // 10**9

        return (
            torch.tensor(x_values, dtype=torch.float32),          # Shape: [input_len]
            torch.tensor(x_time_numeric, dtype=torch.float32),   # Shape: [input_len]
            torch.tensor(x_features, dtype=torch.float32),       # Shape: [input_len, 2]
            torch.tensor(user_id, dtype=torch.long),             # Shape: Scalar
            torch.tensor(y, dtype=torch.float32)                 # Shape: [pred_len]
        )


# Transformer Model with Shape Comments and Strong Typing
class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, time_dim: int, d_model: int, nhead: int,
                 num_layers: int, num_users: int, embedding_dim: int = 16, pred_len: int = 5, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model

        value_dim = d_model // 2
        time_dim_adjusted = d_model // 4
        user_dim = d_model - (value_dim + time_dim_adjusted)

        #  Embedding layers
        self.value_embedding = nn.Linear(input_dim, value_dim)           # Input shape: [batch_size, seq_len, 1]
        self.time_embedding = nn.Linear(time_dim, time_dim_adjusted)     # Input shape: [batch_size, seq_len, 2]
        self.user_embedding = nn.Embedding(num_users, embedding_dim)     # Input shape: [batch_size]
        self.user_projection = nn.Linear(embedding_dim, user_dim)

        # Transformer architecture
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(d_model, pred_len)  # Output layer: Shape [batch_size, pred_len]

    def forward(self, x_values: torch.Tensor, x_time: torch.Tensor, x_features: torch.Tensor, user_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_values: Shape [batch_size, input_len]
            x_time: Shape [batch_size, input_len]
            x_features: Shape [batch_size, input_len, 2]
            user_id: Shape [batch_size]

        Returns:
            predictions: Shape [batch_size, pred_len]
        """
        batch_size, seq_len = x_values.shape
        value_embed = self.value_embedding(x_values.unsqueeze(-1))  # Shape: [batch_size, seq_len, value_dim]
        time_embed = self.time_embedding(x_features)                # Shape: [batch_size, seq_len, time_dim_adjusted]
        user_embed = self.user_projection(self.user_embedding(user_id)).unsqueeze(1).repeat(1, seq_len, 1)  # Shape: [batch_size, seq_len, user_dim]

        x = torch.cat((value_embed, time_embed, user_embed), dim=-1)  # Shape: [batch_size, seq_len, d_model]
        pos_encoding = self.generate_positional_encoding(seq_len).to(x.device)
        x += pos_encoding[:, :seq_len, :]

        tgt = torch.zeros_like(x[:, :1, :]).to(x.device)  # Initial token (e.g., start token)
        tgt = torch.cat([tgt, x[:, :-1, :]], dim=1)  # Shifted target sequence

        # Create target mask (causal mask for autoregressive prediction)
        tgt_mask = self.transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        # ------------------------------------------------------------------- #

        # Pass through Transformer with proper target input
        output = self.transformer(
            src=x,
            tgt=tgt,
            tgt_mask=tgt_mask
        )  # [batch, seq_len, d_model]

        # Predict next sequence (use last output step for forecasting)
        return self.fc_out(output[:, -1, :])  # [batch, pred_len]

    def generate_positional_encoding(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0)  # Shape: [1, seq_len, d_model]

def save_loss_to_csv(train_losses: List[float], val_losses: List[float], output_dir: str = "/home/ioana/Desktop/Model_Licenta/output2") -> None:
    """
    Save training and validation losses per epoch into a CSV file named by the current date and time.

    Args:
        train_losses (List[float]): List of training losses per epoch.
        val_losses (List[float]): List of validation losses per epoch.
        output_dir (str): Directory to save the CSV file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Current datetime for file naming
    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"loss_log_{run_time}.csv"
    file_path = os.path.join(output_dir, filename)
    print(f" train loss: {train_losses[0]:.4f} | val loss: {val_losses[0]:.4f}")
    print(f"length of train loss: {len(train_losses)}")
    print(f"length of validation loss: {len(val_losses)}")

    min_length = min(len(train_losses), len(val_losses))
    train_losses = train_losses[:min_length]
    val_losses = val_losses[:min_length]
    print(f" Using minimum length: {min_length}")

    # Create DataFrame and save to CSV
    loss_df = pd.DataFrame({
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    loss_df.to_csv(file_path, index=False)
    print(f" Training and validation losses saved to: {file_path}")

#  Function: Save Predictions with Denormalization
def save_predictions2_to_csv(user_ids: List[int], times: List[str], y_true: List[float],
                             y_pred: List[float], min_value: float, max_value: float,
                             filename: str = "predictions_transfer_learning.csv") -> None:
    """
    Save predictions to CSV with denormalized real and predicted values.
    """
    print(f"Length of user_ids: {len(user_ids)}, times: {len(times)}, y_true: {len(y_true)}, y_pred: {len(y_pred)}")

    #  Ensure equal lengths
    min_length = min(len(user_ids), len(times), len(y_true), len(y_pred))
    user_ids, times, y_true, y_pred = user_ids[:min_length], times[:min_length], y_true[:min_length], y_pred[:min_length]

    #  Denormalize values
    y_true_denorm = [(val * (max_value - min_value)) + min_value for val in y_true]
    y_pred_denorm = [(val * (max_value - min_value)) + min_value for val in y_pred]

    # Create and save DataFrame
    predictions_df = pd.DataFrame({
        "user_id": user_ids,
        "time": times,
        "real_value": y_true_denorm,
        "predicted_value": y_pred_denorm
    })

    output_dir = '/home/ioana/Desktop/Model_Licenta/output2'
    os.makedirs(output_dir, exist_ok=True)
    file_path_pred = os.path.join(output_dir, filename)
    predictions_df.to_csv(file_path_pred, index=False)
    print(f"Predictions saved to: {file_path_pred}")

print(torch.cuda.is_available())
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {device}")

# Main Execution
file_path = "/home/ioana/Desktop/Preprocesare_Date_Licenta/process_pmdata/filter_merged_processed_data_pmdata.csv"
new_data = pd.read_csv(file_path)
new_data["Time"] = pd.to_datetime(new_data["Time"])
user_id_map = {user_id: idx for idx, user_id in enumerate(new_data["Id"].unique())}

min_value, max_value = new_data["Value"].min(), new_data["Value"].max()
with open('/home/ioana/Desktop/Model_Licenta/data/scaler2_min_max.pkl', 'wb') as f:
    pickle.dump((min_value, max_value), f)

input_len, new_pred_len, overlap, batch_size = 30, 10, 20, 64
dataset = HeartRateDataset(new_data, input_len, new_pred_len, overlap, user_id_map)

scaler: MinMaxScaler = MinMaxScaler()
scaler.fit(new_data["Value"].values.reshape(-1, 1))


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

# Load Model & Transfer Learning
checkpoint = torch.load("/home/ioana/Desktop/Model_Licenta/model_jan_one/best_model.pth")
model = TransformerModel(input_dim=1, time_dim=2, d_model=64, nhead=4,
                         num_layers=3, num_users=len(user_id_map), embedding_dim=16,
                         pred_len=new_pred_len, dropout=0.1)
model.to(device)
# model.load_state_dict(checkpoint, strict=False)
#
# # Freeze layers except final output
# for param in model.parameters():
#     param.requires_grad = False
# model.fc_out = nn.Linear(64, new_pred_len).to(device)
# for param in model.fc_out.parameters():
#     param.requires_grad = True

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# Training Loop
train_losses: List[float] = []
val_losses: List[float] = []
epochs = 20
mse_scores, mae_scores, r2_scores = [], [], []

#dimensiunea modelului
# attention head - optimizare / script optimizare optuna ( divizivil cu dim modelul/tensoer care a=e acum 16
# cat de mult crestere dim de intrare si cat de mult prezice
# early stopping si la al doilea
#


for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    total_val_loss = 0

    # Define column names
    column_names = [
        "Time", "Epoch", "Train Loss", "Validation Loss",
        "MAE Norm", "MAE Denorm", "MSE Norm", "MSE Denorm", "R² Norm", "R² Denorm"
    ]
    results_file = "/home/ioana/Desktop/Model_Licenta/output2/training_results.csv"

    # Ensure CSV file exists with headers
    if not os.path.exists(results_file):
        pd.DataFrame(columns=column_names).to_csv(results_file, index=False)



    for x_values, x_time, x_features, user_id, y in train_loader:
        x_values, x_time, x_features, user_id, y = (x_values.to(device), x_time.to(device),
                                                    x_features.to(device), user_id.to(device), y.to(device))

        optimizer.zero_grad()
        predictions = model(x_values, x_time, x_features, user_id)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

    model.eval()
    total_val_loss = 0
    y_true, y_pred, times, user_ids_list = [], [], [], []

    with torch.no_grad():
        for x_values, x_time, x_features, user_id, y in val_loader:
            x_values, x_time, x_features, user_id, y = (x_values.to(device), x_time.to(device),
                                                        x_features.to(device), user_id.to(device), y.to(device))
            predictions = model(x_values, x_time, x_features, user_id)
            loss = criterion(predictions, y)
            total_val_loss += loss.item()

            y_true_denorm = (y.cpu().numpy().flatten() * (max_value - min_value)) + min_value
            y_pred_denorm = (predictions.cpu().numpy().flatten() * (max_value - min_value)) + min_value

            batch_size, pred_len = predictions.shape
            times_batch = x_time[:, -1].cpu().numpy()  # Last timestamp per sequence
            times_batch = np.repeat(times_batch, pred_len)  # Repeat per predicted step
            times_batch = pd.to_datetime(times_batch, unit='s', origin='unix').astype(str).tolist()

            user_ids_batch = np.repeat(user_id.cpu().numpy().flatten(), pred_len)
            user_ids_list.extend(user_ids_batch.tolist())

            y_true.extend(y_true_denorm.tolist())
            y_pred.extend(y_pred_denorm.tolist())
            times.extend(times_batch)

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
    new_row = pd.DataFrame([[timestamp, epoch + 1, avg_train_loss, avg_val_loss, mae_denorm, mse_denorm, r2_denorm]],
                           columns=["Time", "Epoch", "Train Loss", "Validation Loss", "MAE", "MSE", "R²"])
    new_row.to_csv(results_file, mode='a', header=False, index=False)

    print(
        f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | MAE denorm: {mae_denorm:.4f} | MSE denorm: {mse_denorm:.4f} | R² denorm: {r2_denorm:.4f}"
    )
    print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")

torch.save(model.state_dict(), "best_model_transfer_learning.pth")

def save_loss_to_csv(train_losses, val_losses, filename="/home/ioana/Desktop/Model_Licenta/output2/train_val_losses.csv"):
    df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    df.to_csv(filename, index=False)

save_loss_to_csv(train_losses, val_losses)

y_true_all, y_pred_all, times_all, user_ids_all = [], [], [], []
model.eval()
with torch.no_grad():
    for x_values, x_time, x_features, user_id, y in val_loader:
        x_values, x_time, x_features, user_id, y = (x_values.to(device), x_time.to(device),
                                                    x_features.to(device), user_id.to(device), y.to(device))
        predictions = model(x_values, x_time, x_features, user_id)
        y_true_all.extend(y.cpu().numpy().flatten().tolist())
        y_pred_all.extend(predictions.cpu().numpy().flatten().tolist())
        times_batch = pd.to_datetime(x_time[:, -1].cpu().numpy(), unit='s', origin='unix', utc=True).astype(str).tolist()
        times_all.extend(times_batch)
        user_ids_all.extend(user_id.cpu().numpy().flatten().tolist())

save_predictions2_to_csv(user_ids_all, times_all, y_true_all, y_pred_all, min_value, max_value, filename="/home/ioana/Desktop/Model_Licenta/output2/predictions_transfer_learning.csv")
