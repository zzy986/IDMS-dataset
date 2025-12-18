import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset

from torch.utils.data import DataLoader, TensorDataset

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
x_data_2023 = pd.read_csv('Sensor_2023_daytime.csv')
y_data_2023 = pd.read_csv('Inverter_2023_daytime.csv')
x_data_2022 = pd.read_csv('Sensor_2022_daytime.csv')
y_data_2022 = pd.read_csv('Inverter_2022_daytime.csv')

# Combine data
x_data_combined = pd.concat([x_data_2022, x_data_2023], axis=0, ignore_index=True)
y_data_combined = pd.concat([y_data_2022, y_data_2023], axis=0, ignore_index=True)

# Prepare input features
X = np.stack([x_data_combined['Module_Temperature_degF'].values,
              x_data_combined['Ambient_Temperature_degF'].values,
              x_data_combined['Solar_Irradiation_Wpm2'].values,
              x_data_combined['Wind_Speed_mps'].values], axis=1)

# Prepare targets (Voltage and Current)
y_voltage = y_data_combined['DC_Voltage'].values.reshape(-1, 1)
y_current = y_data_combined['DC_Current'].values.reshape(-1, 1)

# Standardize features and targets
scaler_X = StandardScaler()
scaler_y_voltage = StandardScaler()
scaler_y_current = StandardScaler()

X = scaler_X.fit_transform(X)
y_voltage = scaler_y_voltage.fit_transform(y_voltage)
y_current = scaler_y_current.fit_transform(y_current)

# Split the data into training, validation, and test sets
n_samples = X.shape[0]
train_size = int(n_samples * 0.8)

X_train = X[:train_size]
y_train_voltage = y_voltage[:train_size]
y_train_current = y_current[:train_size]

X_temp = X[train_size:]
y_temp_voltage = y_voltage[train_size:]
y_temp_current = y_current[train_size:]

n_temp_samples = X_temp.shape[0]
valid_size = int(n_temp_samples * 0.5)

X_valid = X_temp[:valid_size]
y_valid_voltage = y_temp_voltage[:valid_size]
y_valid_current = y_temp_current[:valid_size]

X_test = X_temp[valid_size:]
y_test_voltage = y_temp_voltage[valid_size:]
y_test_current = y_temp_current[valid_size:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)

y_train_voltage_tensor = torch.tensor(y_train_voltage, dtype=torch.float32).to(device)
y_valid_voltage_tensor = torch.tensor(y_valid_voltage, dtype=torch.float32).to(device)
y_test_voltage_tensor = torch.tensor(y_test_voltage, dtype=torch.float32).to(device)

y_train_current_tensor = torch.tensor(y_train_current, dtype=torch.float32).to(device)
y_valid_current_tensor = torch.tensor(y_valid_current, dtype=torch.float32).to(device)
y_test_current_tensor = torch.tensor(y_test_current, dtype=torch.float32).to(device)

# Create separate datasets for voltage and current
train_dataset_voltage = TensorDataset(X_train_tensor, y_train_voltage_tensor)
valid_dataset_voltage = TensorDataset(X_valid_tensor, y_valid_voltage_tensor)
test_dataset_voltage = TensorDataset(X_test_tensor, y_test_voltage_tensor)

train_dataset_current = TensorDataset(X_train_tensor, y_train_current_tensor)
valid_dataset_current = TensorDataset(X_valid_tensor, y_valid_current_tensor)
test_dataset_current = TensorDataset(X_test_tensor, y_test_current_tensor)

# Create DataLoaders for voltage and current
batch_size = 256
train_loader_voltage = DataLoader(train_dataset_voltage, batch_size=batch_size, shuffle=True)
valid_loader_voltage = DataLoader(valid_dataset_voltage, batch_size=batch_size, shuffle=False)
test_loader_voltage = DataLoader(test_dataset_voltage, batch_size=batch_size, shuffle=False)

train_loader_current = DataLoader(train_dataset_current, batch_size=batch_size, shuffle=True)
valid_loader_current = DataLoader(valid_dataset_current, batch_size=batch_size, shuffle=False)
test_loader_current = DataLoader(test_dataset_current, batch_size=batch_size, shuffle=False)

print(f"Voltage Training set size: {len(train_loader_voltage.dataset)} samples")
print(f"Current Training set size: {len(train_loader_current.dataset)} samples")

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.5):
        super(LSTMRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)  # 确保输出层的维度与目标数据一致

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 输出维度应为 (batch_size, output_dim)
        return out



# Model definitions
models_voltage = {
    'LSTM': LSTMRegressor(input_dim=X_train_tensor.shape[2], hidden_dim=128, output_dim=1).to(device),
}

models_current = {
    'LSTM': LSTMRegressor(input_dim=X_train_tensor.shape[2], hidden_dim=128, output_dim=1).to(device),
}

epochs = 100

# Training and evaluation function (modified to handle single output)
def train_and_evaluate_model(model, train_loader, valid_loader, test_loader, optimizer, criterion, epochs=epochs):
    train_losses = []
    valid_losses = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_valid_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                valid_predictions = model(X_batch)
                valid_loss = criterion(valid_predictions, y_batch)
                epoch_valid_loss += valid_loss.item()

        valid_losses.append(epoch_valid_loss / len(valid_loader))
        scheduler.step(epoch_valid_loss)

    # 测试集评估
    model.eval()
    test_loss = 0
    test_predictions = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            test_loss += loss.item()
            test_predictions.append(predictions.cpu().numpy())

    test_loss /= len(test_loader)
    test_predictions = np.vstack(test_predictions)
    return train_losses, valid_losses, test_loss, test_predictions

# Training and evaluating each model separately for voltage and current
results_voltage = {}
results_current = {}

# Voltage training
for model_name, model in models_voltage.items():
    print(f"\nTraining {model_name} for Voltage...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
    criterion = torch.nn.MSELoss()

    train_losses, valid_losses, test_loss, test_predictions = train_and_evaluate_model(
        model, train_loader_voltage, valid_loader_voltage, test_loader_voltage, optimizer, criterion, epochs=epochs
    )

    results_voltage[model_name] = {
        'Train Loss': train_losses,
        'Valid Loss': valid_losses,
        'Test Loss': test_loss,
        'Predictions': test_predictions
    }

    print(f"{model_name} Voltage Test Loss: {test_loss:.4f}")

# Current training
for model_name, model in models_current.items():
    print(f"\nTraining {model_name} for Current...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    train_losses, valid_losses, test_loss, test_predictions = train_and_evaluate_model(
        model, train_loader_current, valid_loader_current, test_loader_current, optimizer, criterion, epochs=epochs
    )

    results_current[model_name] = {
        'Train Loss': train_losses,
        'Valid Loss': valid_losses,
        'Test Loss': test_loss,
        'Predictions': test_predictions
    }

    print(f"{model_name} Current Test Loss: {test_loss:.4f}")

# Plotting loss curves for voltage
plt.figure(figsize=(12, 6))
for model_name, result in results_voltage.items():
    plt.plot(result['Train Loss'], label=f'{model_name} Train Loss Voltage')
    plt.plot(result['Valid Loss'], label=f'{model_name} Valid Loss Voltage', linestyle='--')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for Voltage')
plt.legend()
plt.show()

# Plotting loss curves for current
plt.figure(figsize=(12, 6))
for model_name, result in results_current.items():
    plt.plot(result['Train Loss'], label=f'{model_name} Train Loss Current')
    plt.plot(result['Valid Loss'], label=f'{model_name} Valid Loss Current', linestyle='--')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for Current')
plt.legend()
plt.show()

# Inverse transform predictions for voltage
def inverse_transform(scaler, data):
    return scaler.inverse_transform(data)

actual_voltage = inverse_transform(scaler_y_voltage, y_test_voltage_tensor.cpu().numpy())
predicted_voltage = inverse_transform(scaler_y_voltage, results_voltage['LSTM']['Predictions'])

# Inverse transform predictions for current
actual_current = inverse_transform(scaler_y_current, y_test_current_tensor.cpu().numpy())
predicted_current = inverse_transform(scaler_y_current, results_current['LSTM']['Predictions'])

# Compute metrics for voltage
mae_voltage = mean_absolute_error(actual_voltage, predicted_voltage)
rmse_voltage = mean_squared_error(actual_voltage, predicted_voltage, squared=False)

# Compute metrics for current
mae_current = mean_absolute_error(actual_current, predicted_current)
rmse_current = mean_squared_error(actual_current, predicted_current, squared=False)


def calculate_smape(y_true, y_pred):
    epsilon = 1e-10  # 防止除零
    numerator = torch.abs(y_true - y_pred)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2 + epsilon  # 避免分母为零
    smape = torch.mean(numerator / denominator) * 100
    return smape.item()

# Inverse transform predictions for voltage and current
actual_voltage = inverse_transform(scaler_y_voltage, y_test_voltage_tensor.cpu().numpy())
predicted_voltage = inverse_transform(scaler_y_voltage, results_voltage['LSTM']['Predictions'])

actual_current = inverse_transform(scaler_y_current, y_test_current_tensor.cpu().numpy())
predicted_current = inverse_transform(scaler_y_current, results_current['LSTM']['Predictions'])

# Compute metrics for voltage
mse_voltage = mean_squared_error(actual_voltage, predicted_voltage)
r2_voltage = r2_score(actual_voltage, predicted_voltage)
smape_voltage = calculate_smape(torch.tensor(actual_voltage), torch.tensor(predicted_voltage))

# Compute metrics for current
mse_current = mean_squared_error(actual_current, predicted_current)
r2_current = r2_score(actual_current, predicted_current)
smape_current = calculate_smape(torch.tensor(actual_current), torch.tensor(predicted_current))

# Output metrics for voltage
print(f"Voltage MSE: {mse_voltage:.4f}, R²: {r2_voltage:.4f}, SMAPE: {smape_voltage:.4f}%")

# Output metrics for current
print(f"Current MSE: {mse_current:.4f}, R²: {r2_current:.4f}, SMAPE: {smape_current:.4f}%")


print(f"Voltage MAE: {mae_voltage:.4f}, Voltage RMSE: {rmse_voltage:.4f}")
print(f"Current MAE: {mae_current:.4f}, Current RMSE: {rmse_current:.4f}")

# Plot actual vs predicted voltage
plt.figure(figsize=(15, 6))
plt.plot(actual_voltage[0:1000], label='Actual Voltage')
plt.plot(predicted_voltage[0:1000], label='Predicted Voltage', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Voltage')
plt.title('Actual vs Predicted Voltage Over Time')
plt.legend()
plt.show()

# Plot actual vs predicted current
plt.figure(figsize=(15, 6))
plt.plot(actual_current[0:1000], label='Actual Current')
plt.plot(predicted_current[0:1000], label='Predicted Current', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Current')
plt.title('Actual vs Predicted Current Over Time')
plt.legend()
plt.show()

