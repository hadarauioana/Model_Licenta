import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler


# Heart Rate Dataset Class
class HeartRateDataset(Dataset):
    def __init__(self, df, input_len, pred_len, overlap, user_id_map):
        self.data = []
        self.input_len = input_len
        self.pred_len = pred_len
        self.overlap = overlap
        self.user_id_map = user_id_map

        scaler = MinMaxScaler()
        for patient_id, group in df.groupby("Id"):
            group = group.sort_values("Time").reset_index(drop=True)
            values = scaler.fit_transform(group["Value"].values.reshape(-1, 1)).flatten()

            group["Hour"] = group["Time"].dt.hour / 23.0
            group["DayOfWeek"] = group["Time"].dt.dayofweek / 6.0
            time_features = group[["Hour", "DayOfWeek"]].values

            step = input_len - overlap
            for i in range(0, len(values) - input_len - pred_len + 1, step):
                x_values = values[i:i + input_len]
                x_time = time_features[i:i + input_len]
                y = values[i + input_len:i + input_len + pred_len]
                user_id = self.user_id_map[patient_id]
                self.data.append((x_values, x_time, user_id, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_values, x_time, user_id, y = self.data[idx]
        return (
            torch.tensor(x_values, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(x_time, dtype=torch.float32),
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32)
        )

# LSTM Model for Heart Rate Prediction
class LSTMModel(nn.Module):
    def __init__(self, input_dim, time_dim, hidden_dim, num_layers, num_users, embedding_dim=16, pred_len=5, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.value_embedding = nn.Linear(input_dim, hidden_dim // 2)
        self.time_embedding = nn.Linear(time_dim, hidden_dim // 4)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_projection = nn.Linear(embedding_dim, hidden_dim // 4)

        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, pred_len)

    def forward(self, x_values, x_time, user_id):
        value_embed = self.value_embedding(x_values)
        time_embed = self.time_embedding(x_time)
        user_embed = self.user_projection(self.user_embedding(user_id)).unsqueeze(1).repeat(1, x_values.shape[1], 1)

        x = torch.cat((value_embed, time_embed, user_embed), dim=-1)

        lstm_out, _ = self.lstm(x)
        output = self.fc_out(lstm_out[:, -1, :])
        return output

# Load and Preprocess Data
file_path = "heartrate_data.csv"
data = pd.read_csv(file_path)
data["Time"] = pd.to_datetime(data["Time"])

user_ids = data["Id"].unique()
user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}

# Hyperparameters
input_len = 15
pred_len = 5
overlap = 5
batch_size = 64
hidden_dim = 64
num_layers = 2

# Create Dataset and DataLoader
dataset = HeartRateDataset(data, input_len, pred_len, overlap, user_id_map)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

# Initialize Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(
    input_dim=1, time_dim=2, hidden_dim=hidden_dim, num_layers=num_layers,
    num_users=len(user_ids), embedding_dim=16, pred_len=pred_len, dropout=0.1
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=5):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x_values, x_time, user_id, y in train_loader:
            x_values, x_time, user_id, y = x_values.to(device), x_time.to(device), user_id.to(device), y.to(device)

            optimizer.zero_grad()
            predictions = model(x_values, x_time, user_id)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_values, x_time, user_id, y in val_loader:
                x_values, x_time, user_id, y = x_values.to(device), x_time.to(device), user_id.to(device), y.to(device)
                predictions = model(x_values, x_time, user_id)
                loss = criterion(predictions, y)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

# Train the Model
train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=5)

# Save the Model
torch.save(model.state_dict(), "lstm_heart_rate_model.pth")
print("Model saved successfully!")
