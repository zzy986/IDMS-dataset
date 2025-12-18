from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# 读取数据
x_data_2023 = pd.read_csv('Sensor_2023_daytime.csv')
y_data_2023 = pd.read_csv('Inverter_2023_daytime.csv')
x_data_2022 = pd.read_csv('Sensor_2022_daytime_1.csv')
y_data_2022 = pd.read_csv('Inverter_2022_daytime_1.csv')

# 合并数据
x_data_combined = pd.concat([x_data_2023,], axis=0, ignore_index=True)
y_data_combined = pd.concat([y_data_2023], axis=0, ignore_index=True)

# 数据检查
print(x_data_combined.head())
print(x_data_combined.tail())

# 准备特征和目标变量
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
train_size = int(n_samples * 0.6)
temp_size = n_samples - train_size

X_train = X[:train_size]
y_train = y[:train_size]

X_temp = X[train_size:]
y_temp = y[train_size:]

n_temp_samples = X_temp.shape[0]
valid_size = int(n_temp_samples * 0.5)

X_valid = X_temp[:valid_size]
y_valid = y_temp[:valid_size]

X_test = X_temp[valid_size:]
y_test = y_temp[valid_size:]

# 定义多输出SVR模型
svm_model = MultiOutputRegressor(SVR(kernel='rbf', C=50.0, epsilon=0.05))

# 训练模型
start_time = time.time()
svm_model.fit(X_train, y_train)
end_time = time.time()
training_time = end_time - start_time
print(f"Training Time for MultiOutput SVM Model: {training_time:.4f} seconds")

# 预测验证集和测试集
y_valid_pred = svm_model.predict(X_valid)
y_test_pred = svm_model.predict(X_test)

# 逆变换到原始尺度
y_test_true = scaler_y.inverse_transform(y_test)
y_test_pred_inverse = scaler_y.inverse_transform(y_test_pred)

# 计算性能指标
mae = mean_absolute_error(y_test_true, y_test_pred_inverse)
mse = mean_squared_error(y_test_true, y_test_pred_inverse)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_true, y_test_pred_inverse)

print("\nMultiOutput SVM Model Performance on Test Data:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")

# 可视化实际值与预测值的对比
plt.figure(figsize=(12, 5))

# 电压预测对比
plt.subplot(1, 2, 1)
plt.scatter(y_test_true[:, 0], y_test_pred_inverse[:, 0], alpha=0.5, label='Predicted')
plt.plot([y_test_true[:, 0].min(), y_test_true[:, 0].max()],
         [y_test_true[:, 0].min(), y_test_true[:, 0].max()], 'r--', label='Ideal')
plt.xlabel('True Voltage (V)')
plt.ylabel('Predicted Voltage (V)')
plt.title('MultiOutput SVM - Voltage (V) Prediction')
plt.legend()

# 电流预测对比
plt.subplot(1, 2, 2)
plt.scatter(y_test_true[:, 1], y_test_pred_inverse[:, 1], alpha=0.5, label='Predicted')
plt.plot([y_test_true[:, 1].min(), y_test_true[:, 1].max()],
         [y_test_true[:, 1].min(), y_test_true[:, 1].max()], 'r--', label='Ideal')
plt.xlabel('True Current (I)')
plt.ylabel('Predicted Current (I)')
plt.title('MultiOutput SVM - Current (I) Prediction')
plt.legend()

plt.tight_layout()
#plt.savefig('multioutput_svm_scatter.png')
plt.show()

plt.figure(figsize=(12, 10))

# 电压预测的时间序列对比
plt.subplot(2, 1, 1)
plt.plot(y_test_true[0:1000, 0], label='True Voltage', alpha=0.7)
plt.plot(y_test_pred_inverse[0:1000, 0], label='Predicted Voltage', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Voltage (V)')
plt.title('Actual vs Predicted Voltage (MultiOutput SVM)')
plt.legend()

# 电流预测的时间序列对比
plt.subplot(2, 1, 2)
plt.plot(y_test_true[0:1000, 1], label='True Current', alpha=0.7)
plt.plot(y_test_pred_inverse[0:1000, 1], label='Predicted Current', alpha=0.7)
plt.xlabel('Sample Index')
plt.ylabel('Current (I)')
plt.title('Actual vs Predicted Current (MultiOutput SVM)')
plt.legend()

plt.tight_layout()
#plt.savefig('multioutput_svm_series.png')
plt.show()