import pandas as pd
import numpy as np
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler

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

        # Initialize Min-Max scaler
        scaler = MinMaxScaler()

        # Process each patient group
        for patient_id, group in df.groupby("Id"):
            # Sort group by timestamp and normalize heart rate values
            group = group.sort_values("Time").reset_index(drop=True)
            values = scaler.fit_transform(group["Value"].values.reshape(-1, 1)).flatten()

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
                y = values[i + input_len:i + input_len + pred_len]  # Shape: [pred_len]
                user_id = self.user_id_map[patient_id]  # Single scalar
                self.data.append((x_values, x_time, x_features, user_id, y))

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
        x_values, x_time, x_features, user_id, y = self.data[idx]

        # Convert datetime64 to seconds since epoch
        x_time_numeric = (x_time.astype('datetime64[s]') - np.datetime64('1970-01-01T00:00:00Z')).astype('int')

        return (
            torch.tensor(x_values, dtype=torch.float32),  # [input_len]
            torch.tensor(x_time_numeric, dtype=torch.float32),  # [input_len]
            torch.tensor(x_features, dtype=torch.float32),  # [input_len, 2]
            torch.tensor(user_id, dtype=torch.long),  # Scalar
            torch.tensor(y, dtype=torch.float32)  # [pred_len]
        )

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, time_dim: int, d_model: int, nhead: int, num_layers: int, num_users: int,
                 embedding_dim: int = 16, pred_len: int = 5, dropout: float = 0.1):
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

        # Calculate dimensions for embeddings
        value_dim = d_model // 2  # Larger portion for values
        time_dim_adjusted = d_model // 4  # Smaller portion for time features
        user_dim = d_model - (value_dim + time_dim_adjusted)  # Remaining portion for user embedding

        # Define embedding layers
        self.value_embedding = nn.Linear(input_dim, value_dim)  # [batch, seq_len, value_dim]
        self.time_embedding = nn.Linear(time_dim, time_dim_adjusted)  # [batch, seq_len, time_dim_adjusted]
        self.user_embedding = nn.Embedding(num_users, embedding_dim)  # [batch, embedding_dim]
        self.user_projection = nn.Linear(embedding_dim, user_dim)  # [batch, user_dim]

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

    def forward(self, x_values: torch.Tensor, x_time: torch.Tensor, x_features: torch.Tensor, user_id: torch.Tensor):
        """
        Forward pass for the model.
        Parameters:
        - x_values: Normalized input values. Shape: [batch, seq_len, input_dim]
        - x_time: Numeric timestamps. Shape: [batch, seq_len]
        - x_features: Temporal features. Shape: [batch, seq_len, time_dim]
        - user_id: User ID indices. Shape: [batch]

        Returns: Predicted values. Shape: [batch, pred_len]
        """
        batch_size, seq_len = x_values.shape

        # Compute embeddings
        value_embed = self.value_embedding(x_values.unsqueeze(-1))  # [batch, seq_len, value_dim]
        time_embed = self.time_embedding(x_features)  # [batch, seq_len, time_dim_adjusted]
        user_embed = self.user_projection(self.user_embedding(user_id)).unsqueeze(1).repeat(1, seq_len, 1)  # [batch, seq_len, user_dim]

        # Concatenate embeddings
        x = torch.cat((value_embed, time_embed, user_embed), dim=-1)  # [batch, seq_len, d_model]

        # Add positional encoding
        pos_encoding = self.generate_positional_encoding(seq_len).to(x.device)  # [1, seq_len, d_model]
        x += pos_encoding[:, :seq_len, :]

        # Pass through Transformer
        output = self.transformer(x, x)  # [batch, seq_len, d_model]

        # Predict next sequence
        return self.fc_out(output[:, -1, :])  # [batch, pred_len]




































































































