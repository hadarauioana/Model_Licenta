import pickle

import pandas as pd
import numpy as np
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


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

        # Process each patient group
        for patient_id, group in df.groupby("Id"):
            # Sort group by timestamp and normalize heart rate values
            group = group.sort_values("Time").reset_index(drop=True)
            #values = scaler.fit_transform(group["Value"].values.reshape(-1, 1)).flatten()
            values = (group["Value"] - min_value) / (max_value - min_value)
            values = values.to_numpy()  # Convert to numpy array if needed

            # Create lag features
            group["lag_1"] = group["Value"].shift(1)  # Lag of 1 minute
            group["lag_5"] = group["Value"].shift(5)  # Lag of 5 minutes
            lag1 = (group["lag_1"] - min_value) / (max_value - min_value)
            lag5 = (group["lag_5"] - min_value) / (max_value - min_value)
            # lags_features = group[[lag1, lag5]].values  # Shape: [num_samples, 2]
            lag1 = lag1.to_numpy()
            lag5 = lag5.to_numpy()


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
                x_lag5 = lag5[i:i + input_len]

                y = values[i + input_len:i + input_len + pred_len]  # Shape: [pred_len]
                user_id = self.user_id_map[patient_id]  # Single scalar
                self.data.append((x_values, x_time, x_features,x_lag1,x_lag5, user_id, y))

        # Process each patient group
        # for patient_id, group in df.groupby("Id"):
        #     # Sort group by timestamp
        #     group = group.sort_values("Time").reset_index(drop=True)
        #
        #     # Ensure numeric user_id mapping
        #     user_id = self.user_id_map.get(patient_id, -1)

            # Find continuous segments where there are at least 60 consecutive minutes
            # group["TimeDiff"] = group["Time"].diff().dt.total_seconds().fillna(180)
            # group["Segment"] = (group["TimeDiff"] > 180).cumsum()

            # df_resampled = df.set_index("Time").resample("1T").mean()
            # # Step 4: Apply Exponential Moving Average (EMA) to fill missing values
            # df_resampled["Value"] = df_resampled["Value"].ewm(span=5, adjust=False).mean()
            # # Step 5: Reset index to make 'Time' a column again
            # df_resampled = df_resampled.reset_index()
            #
            # # Step 6: Merge Segmentation Back (keep original time-based gaps)
            # df_final = df_resampled.merge(group[["Time", "Segment"]], on="Time", how="left").fillna(method="ffill")

            # for _, segment in group.groupby("Segment"):
            #     # if len(segment) < 40:
            #     #     continue  # Skip segments that are too short to generate at least 2 windows
            #
            #     # Normalize heart rate values
            #     values = scaler.fit_transform(segment["Value"].values.reshape(-1, 1)).flatten()
            #
            #     # Extract real timestamps and temporal features
            #     timestamps = segment["Time"].values
            #     segment["Hour"] = segment["Time"].dt.hour / 23.0  # Normalize hour to [0, 1]
            #     segment["DayOfWeek"] = segment["Time"].dt.dayofweek / 6.0  # Normalize day of week to [0, 1]
            #     time_features = segment[["Hour", "DayOfWeek"]].values
            #
            #     # Define time-based sliding window step
            #     step = 60  # Move by 60 minutes
            #
            #     # Iterate over windows
            #     start_idx = 0
            #     while start_idx + input_len + pred_len <= len(values):
            #         x_values = values[start_idx:start_idx + input_len]  # [input_len]
            #         x_time = timestamps[start_idx:start_idx + input_len]  # [input_len]
            #         x_features = time_features[start_idx:start_idx + input_len]  # [input_len, 2]
            #         y = values[start_idx + input_len:start_idx + input_len + pred_len]  # [pred_len]
            #
            #         # Append data
            #         self.data.append((x_values, x_time, x_features, user_id, y))
            #
            #         # Move the window by 60 minutes
            #         start_idx += step

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
        x_values, x_time, x_features,x_lag1,x_lag5, user_id, y = self.data[idx]

        # Convert datetime64 to seconds since epoch
        x_time_numeric = (x_time.astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')).astype('int')

        return (
            torch.tensor(x_values, dtype=torch.float32),  # [input_len]
            torch.tensor(x_time_numeric, dtype=torch.float32),  # [input_len]
            torch.tensor(x_features, dtype=torch.float32),  # [input_len, 2]
            torch.tensor(x_lag1, dtype=torch.float32),  # [input_len]
            torch.tensor(x_lag5, dtype=torch.float32),  # [input_len]
            torch.tensor(user_id, dtype=torch.long),  # Scalar
            torch.tensor(y, dtype=torch.float32)  # [pred_len]
        )

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, time_dim: int, d_model: int, nhead: int, num_layers: int, num_users: int,
                 embedding_dim: int = 16, pred_len: int = 10, dropout: float = 0.1):
        """
        Parameters:
        - input_dim: Dimensionality of input values (e.g., 1 for univariate time series).
        - time_dim: Dimensionality of time features (e.g., 2 for [hour, day_of_week]).
        - d_model: Dimensionality of Transformer embeddings.
        - nhead: Number of attention heads.
        - num_layers: Number of Transformer encoder/decoder layers.
        - num_users: Number of unique users for embedding.
        - embedding_dim: Dimensionality of user embedding.
        - pred_len: Length of output prediction sequence.
        - dropout: Dropout rate for Transformer.
        """
        super(TransformerModel, self).__init__()
        self.d_model = d_model

        # dmodel =128
        #value di  =64
        # Calculate dimensions for embeddings - init
        value_dim = d_model // 2  # Larger portion for values

        time_dim_adjusted = d_model // 8  # Smaller portion for time features

        # Define embedding layers
        self.value_embedding = nn.Linear(input_dim, value_dim)  # [batch, seq_len, value_dim]
        self.lag1_embedding = nn.Linear(input_dim, embedding_dim)  # [batch, seq_len, value_dim]
        self.lag5_embedding = nn.Linear(input_dim, embedding_dim)  # [batch, seq_len, value_dim]

        # Token Embedding for input values
        # self.value_embedding = TokenEmbedding(1, d_model // 2)  # Use convolution-based embedding

        self.time_embedding = nn.Linear(time_dim, embedding_dim)  # [batch, seq_len, time_dim_adjusted]
        self.user_embedding = nn.Embedding(num_users, embedding_dim)  # [batch, embedding_dim]

        # Define Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, pred_len)  # Final prediction layer [batch, pred_len]

    def generate_positional_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Generates sinusoidal positional encoding for the sequence.
        Returns: [1, seq_len, d_model]
        """
        positions = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0)

    def forward(self, x_values: torch.Tensor, x_time: torch.Tensor, x_features: torch.Tensor,x_lag1: torch.Tensor,x_lag5: torch.Tensor, user_id: torch.Tensor):
        """
        Forward pass for the model.
        Parameters:
        - x_values: Normalized input values. Shape: [batch, seq_len, input_dim]
        - x_time: Numeric timestamps. Shape: [batch, seq_len]
        - x_features: Temporal features. Shape: [batch, seq_len, time_dim]
        - x_lag1: minuts before.  Shape: [batch, seq_len]
        - user_id: User ID indices. Shape: [batch]

        Returns: Predicted values. Shape: [batch, pred_len]
        """
        batch_size, seq_len = x_values.shape

        # Compute embeddings
        value_embed = self.value_embedding(x_values.unsqueeze(-1))  # [batch, seq_len, value_dim]
        lag1_embed = self.lag1_embedding(x_values.unsqueeze(-1))  # [batch, seq_len, value_dim]
        lag5_embed = self.lag5_embedding(x_values.unsqueeze(-1))  # [batch, seq_len, value_dim]

        time_embed = self.time_embedding(x_features)  # [batch, seq_len, time_dim_adjusted]
        user_embed = self.user_embedding(user_id).unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, user_dim]
        # --adds a sequence dimension to make it [batch, 1, user_dim]                --replicates the user embedding along the sequence dimension so that it can be concatenated with other embeddings, resulting in [batch, seq_len, user_dim].

        # Concatenate embeddings
        x = torch.cat((value_embed, time_embed, user_embed, lag1_embed,lag5_embed), dim=-1)  # [batch, seq_len, d_model]
        # print(f"\n[ENCODER INPUT - x] Shape: {x.shape}")
        # print(f"[ENCODER INPUT - x] Sample Data:\n{x[0, :5, :5]}")  # Print first sequence, first 5 timesteps, first 5 features
        #Combines the three embeddings along the feature dimension. The resulting tensor has shape [batch, seq_len, d_model] because the individual dimensions add up to d_model.

        # Add positional encoding
        pos_encoding = self.generate_positional_encoding(seq_len).to(x.device)  # [1, seq_len, d_model]
        x += pos_encoding[:, :seq_len, :]
        # print(f"\n[ENCODER INPUT + POS ENCODING] Shape: {x.shape}")
        # print(f"[ENCODER INPUT + POS ENCODING] Sample Data:\n{x[0, :5, :5]}")

        # Shift the input by one step for the decoder (typical autoregressive target)
        # tgt starts with a zero tensor or last known value and shifts the rest of the input
        tgt = torch.zeros_like(x[:, :1, :]).to(x.device)  # Initial token (e.g., start token)
        tgt = torch.cat([tgt, x[:, :-1, :]], dim=1)  # Shifted target sequence
        # print(f"\n[DECODER INPUT - tgt] Shape: {tgt.shape}")
        # print(f"[DECODER INPUT - tgt] Sample Data:\n{tgt[0, :5, :5]}")

        # Create target mask (causal mask for autoregressive prediction)
        tgt_mask = self.transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        # ------------------------------------------------------------------- #

        # Pass through Transformer with proper target input
        output = self.transformer(
            src=x,
            tgt=tgt,
            tgt_mask=tgt_mask
        )  # [batch, seq_len, d_model]
        # print(f"\n[TRANSFORMER OUTPUT] Shape: {output.shape}")
        # print(f"[TRANSFORMER OUTPUT] Sample Data:\n{output[0, :5, :5]}")

        # Predict next sequence (use last output step for forecasting)
        final_output = self.fc_out(output[:, -1, :])  # [batch, pred_len]
        # print(f"\n[FINAL OUTPUT] Shape: {final_output.shape}")
        # print(f"[FINAL OUTPUT] Sample Data:\n{final_output[0]}")
        return final_output
























































































