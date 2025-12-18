from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import time
# 初始化SVM模型
svm_model_voltage = SVR(kernel='rbf', C=50.0, epsilon=0.05)  # 用于预测电压的SVM模型
svm_model_current = SVR(kernel='rbf', C=50.0, epsilon=0.05)  # 用于预测电流的SVM模型



x_data_2023= pd.read_csv('Sensor_2023_daytime.csv')

y_data_2023= pd.read_csv('Inverter_2023_daytime.csv')


x_data_2022 = pd.read_csv('Sensor_2022_daytime.csv')
y_data_2022 = pd.read_csv('Inverter_2022_daytime.csv')



x_data_combined = pd.concat([x_data_2022,x_data_2023], axis=0, ignore_index=True)
y_data_combined = pd.concat([y_data_2022, y_data_2023], axis=0, ignore_index=True)


print(x_data_combined.head())  # 检查前几行，应该是2022年的数据
print(x_data_combined.tail())
X = np.stack([x_data_combined['Module_Temperature_degF'].values,
              x_data_combined['Ambient_Temperature_degF'].values,
              x_data_combined['Solar_Irradiation_Wpm2'].values,
              x_data_combined['Wind_Speed_mps'].values], axis=1)

y = np.stack([y_data_combined['DC_Voltage'].values, y_data_combined['DC_Current'].values], axis=1)




# 标准化特征
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# 按照顺序划分数据集
n_samples = X.shape[0]
train_size = int(n_samples * 0.8)  # 前60%为训练集
temp_size = n_samples - train_size  # 后40%为临时集

# 前60%的数据作为训练集
X_train = X[:train_size]
y_train = y[:train_size]

# 后40%的数据作为临时集
X_temp = X[train_size:]
y_temp = y[train_size:]

# 手动划分验证集和测试集
n_temp_samples = X_temp.shape[0]
valid_size = int(n_temp_samples * 0.5)  # 前50%为验证集

# 前50%的临时数据作为验证集
X_valid = X_temp[:valid_size]
y_valid = y_temp[:valid_size]

# 后50%的临时数据作为测试集
X_test = X_temp[valid_size:]
y_test = y_temp[valid_size:]




start_time_voltage = time.time()
svm_model_voltage.fit(X_train, y_train[:, 0])  # Train voltage model
end_time_voltage = time.time()
training_time_voltage = end_time_voltage - start_time_voltage

# Measure training time for current SVM model
start_time_current = time.time()
svm_model_current.fit(X_train, y_train[:, 1])  # Train current model
end_time_current = time.time()
training_time_current = end_time_current - start_time_current

# Display training times
print(f"\nTraining Time for Voltage SVM Model: {training_time_voltage:.4f} seconds")
print(f"Training Time for Current SVM Model: {training_time_current:.4f} seconds")

# Predict on validation set
y_valid_pred_voltage_svm = svm_model_voltage.predict(X_valid)
y_valid_pred_current_svm = svm_model_current.predict(X_valid)

# Predict on test set
y_test_pred_voltage_svm = svm_model_voltage.predict(X_test)
y_test_pred_current_svm = svm_model_current.predict(X_test)

# Inverse transform predictions to original scale
y_test_true_voltage = scaler_y.inverse_transform(y_test)[:, 0]
y_test_true_current = scaler_y.inverse_transform(y_test)[:, 1]
y_test_pred_voltage_svm_inverse = scaler_y.inverse_transform(
    np.column_stack((y_test_pred_voltage_svm, np.zeros_like(y_test_pred_voltage_svm))))[:, 0]
y_test_pred_current_svm_inverse = scaler_y.inverse_transform(
    np.column_stack((np.zeros_like(y_test_pred_current_svm), y_test_pred_current_svm)))[:, 1]

# Calculate performance metrics for voltage
mae_voltage_svm = mean_absolute_error(y_test_true_voltage, y_test_pred_voltage_svm_inverse)
mse_voltage_svm = mean_squared_error(y_test_true_voltage, y_test_pred_voltage_svm_inverse)
rmse_voltage_svm = np.sqrt(mse_voltage_svm)
r2_voltage_svm = r2_score(y_test_true_voltage, y_test_pred_voltage_svm_inverse)

# Calculate performance metrics for current
mae_current_svm = mean_absolute_error(y_test_true_current, y_test_pred_current_svm_inverse)
mse_current_svm = mean_squared_error(y_test_true_current, y_test_pred_current_svm_inverse)
rmse_current_svm = np.sqrt(mse_current_svm)
r2_current_svm = r2_score(y_test_true_current, y_test_pred_current_svm_inverse)

print("\nSVM Model Performance on Test Data (Voltage):")
print(f"MAE: {mae_voltage_svm:.4f}")
print(f"MSE: {mse_voltage_svm:.4f}")
print(f"RMSE: {rmse_voltage_svm:.4f}")
print(f"R^2: {r2_voltage_svm:.4f}")

print("\nSVM Model Performance on Test Data (Current):")
print(f"MAE: {mae_current_svm:.4f}")
print(f"MSE: {mse_current_svm:.4f}")
print(f"RMSE: {rmse_current_svm:.4f}")
print(f"R^2: {r2_current_svm:.4f}")

# Plot predicted vs. actual values for voltage and current
plt.figure(figsize=(12, 5))

# Voltage (V) comparison
plt.subplot(1, 2, 1)
plt.scatter(y_test_true_voltage, y_test_pred_voltage_svm_inverse, alpha=0.5, label='Predicted')
plt.plot([y_test_true_voltage.min(), y_test_true_voltage.max()],
         [y_test_true_voltage.min(), y_test_true_voltage.max()], 'r--', label='Ideal')
plt.xlabel('True Voltage (V)')
plt.ylabel('Predicted Voltage (V)')
plt.title('SVM - Voltage (V) Prediction')
plt.legend()

# Current (I) comparison
plt.subplot(1, 2, 2)
plt.scatter(y_test_true_current, y_test_pred_current_svm_inverse, alpha=0.5, label='Predicted')
plt.plot([y_test_true_current.min(), y_test_true_current.max()],
         [y_test_true_current.min(), y_test_true_current.max()], 'r--', label='Ideal')
plt.xlabel('True Current (I)')
plt.ylabel('Predicted Current (I)')
plt.title('SVM - Current (I) Prediction')
plt.legend()

plt.tight_layout()
plt.savefig('svm_scatter.png')
plt.show()

# Plot actual vs predicted values over time for better visualization
plt.figure(figsize=(12, 10))

# Voltage prediction over time
plt.subplot(2, 1, 1)
plt.plot(y_test_true_voltage[0:1000], label='True Voltage')
plt.plot(y_test_pred_voltage_svm_inverse[0:1000], label='Predicted Voltage', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Voltage (V)')
plt.title('Actual vs Predicted Voltage (SVM)')
plt.legend()

# Current prediction over time
plt.subplot(2, 1, 2)
plt.plot(y_test_true_current[0:1000], label='True Current')
plt.plot(y_test_pred_current_svm_inverse[0:1000], label='Predicted Current', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Current (I)')
plt.title('Actual vs Predicted Current (SVM)')
plt.legend()

plt.tight_layout()
plt.savefig('SVM_series.png')
plt.show()

