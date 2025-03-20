import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import numpy as np



# Dataset class for Heart Rate data
class HeartRateDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_len: int, pred_len: int, overlap: int, user_id_map: dict):
        """
        Parameters:
        - df: Input dataframe, columns = ['Id', 'Time', 'Value'].
        - input_len: Length of input sequence.
        - pred_len: Length of prediction sequence.
        - overlap: Overlap between windows for sliding window approach.
        - user_id_map: Mapping of user IDs to numeric indices.
        """
        self.data = []  # List of tuples containing processed data for each window
        self.input_len = input_len
        self.pred_len = pred_len
        self.overlap = overlap  # Overlap between windows
        self.user_id_map = user_id_map  # Mapping of user IDs to numeric indices

        # engineer feature - lag10, la

        # Initialize Min-Max scaler
        # scaler = MinMaxScaler()
        with open('/home/ioana/Desktop/Model_Licenta/data/scaler_min_max.pkl', 'rb') as f:
            min_value, max_value = pickle.load(f)

        zscore_scaler = StandardScaler()  # Z-score for HR gradients
        # Store processed data
        standardized_gradients = []

        # Process each patient group
        for patient_id, group in df.groupby("Id"):
            # Sort group by timestamp and normalize heart rate values
            group = group.sort_values("Time").reset_index(drop=True)
            #values = scaler.fit_transform(group["Value"].values.reshape(-1, 1)).flatten()
            values = (group["Value"] - min_value) / (max_value - min_value)
            values = values.to_numpy()  # Convert to numpy array if needed

            # Create lag features
            group["lag_1"] = group["Value"].shift(1)  # Lag of 1 minute
            group["gradient"] = group["Value"]-group["Value"].shift(1)

            # **Fix NaNs in lag_1 and gradient before applying transformations**
            group["lag_1"].fillna(method="bfill", inplace=True)  # Backfill NaNs
            group["gradient"].fillna(0, inplace=True)  # Replace NaN gradients with 0

            lag1 = (group["lag_1"] - min_value) / (max_value - min_value)
            # lags_features = group[[lag1, lag5]].values  # Shape: [num_samples, 2]
            lag1 = lag1.to_numpy()

            # Standardization (Z-score) for Gradients
            group["gradient_std"] = zscore_scaler.fit_transform(group[["gradient"]])
            gradients = group["gradient_std"].to_numpy()
            # print("gradients", gradients.shape)
            # print("gradients", gradients)


            # Extract real timestamps and temporal features
            timestamps = group["Time"].values  # Shape: [num_samples]
            group["Hour"] = group["Time"].dt.hour / 23.0  # Normalize hour to [0, 1]
            group["DayOfWeek"] = group["Time"].dt.dayofweek / 6.0  # Normalize day of week to [0, 1]
            time_features = group[["Hour", "DayOfWeek"]].values  # Shape: [num_samples, 2]

            # Define sliding window step size
            step = input_len - overlap

            # Create data windows
            for i in range(0, len(values) - input_len - pred_len + 1, step):
                x_values = values[i:i + input_len]  # Shape: [input_len]
                x_time = timestamps[i:i + input_len]  # Shape: [input_len]
                x_features = time_features[i:i + input_len]  # Shape: [input_len, 2]
                x_lag1 = lag1[i:i + input_len]
                x_gradients = gradients[i:i + input_len]
                # print("xxxx",x_gradients)
                y = values[i + input_len:i + input_len + pred_len]  # Shape: [pred_len]
                user_id = self.user_id_map[patient_id]  # Single scalar
                self.data.append((x_values, x_time, x_features,x_lag1,x_gradients, user_id, y))


    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Returns:
        - x_values: Normalized heart rate values. Shape: [input_len]
        - x_time_numeric: Numeric timestamps. Shape: [input_len]
        - x_features: Temporal features (hour, day of week). Shape: [input_len, 2]
        - user_id: User ID index. Shape: Scalar.
        - y: Future normalized values. Shape: [pred_len]
        """
        x_values, x_time, x_features,x_lag1,x_gradients, user_id, y = self.data[idx]

        # Convert datetime64 to seconds since epoch
        x_time_numeric = (x_time.astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')).astype('int')

        return (
            torch.tensor(x_values, dtype=torch.float32),  # [input_len]
            torch.tensor(x_time_numeric, dtype=torch.float32),  # [input_len]
            torch.tensor(x_features, dtype=torch.float32),  # [input_len, 2]
            torch.tensor(x_lag1, dtype=torch.float32),  # [input_len]
            torch.tensor(x_gradients, dtype=torch.float32),  # [input_len]
            torch.tensor(user_id, dtype=torch.long),  # Scalar
            torch.tensor(y, dtype=torch.float32)  # [pred_len]
        )


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn_weights = nn.Linear(hidden_dim * 2, 1)  # BiLSTM has 2*hidden_dim

    def forward(self, lstm_output):
        attn_scores = self.attn_weights(lstm_output).squeeze(-1)  # Shape: (batch, seq_len)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # Shape: (batch, seq_len, 1)
        context_vector = torch.sum(lstm_output * attn_weights, dim=1)  # Shape: (batch, hidden_dim*2)
        return context_vector, attn_weights


class LSTMBiLSTMAtt(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(LSTMBiLSTMAtt, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # First LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # BiLSTM layer
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

        # Attention layer
        self.attention = Attention(hidden_dim)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x_values, x_time_numeric, x_features, x_lag1, x_gradients):
        # Concatenate additional features
        x = torch.cat((x_values.unsqueeze(-1), x_features, x_lag1.unsqueeze(-1), x_gradients.unsqueeze(-1)), dim=-1)

        lstm_out, _ = self.lstm(x)  # Shape: (batch, seq_len, hidden_dim)
        lstm_out = self.dropout(lstm_out)

        bilstm_out, _ = self.bilstm(lstm_out)  # Shape: (batch, seq_len, hidden_dim*2)
        bilstm_out = self.dropout(bilstm_out)

        context_vector, attn_weights = self.attention(bilstm_out)  # Shape: (batch, hidden_dim*2)

        output = self.fc(context_vector)  # Shape: (batch, 1)
        return output, attn_weights

