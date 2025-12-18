import pandas as pd

# 加载数据
x_data_2023= pd.read_csv('Sensor_2023_daytime.csv')

y_data_2023= pd.read_csv('Inverter_2023_daytime.csv')


x_data_2022 = pd.read_csv('Sensor_2022_daytime.csv')
y_data_2022 = pd.read_csv('Inverter_2022_daytime.csv')

# 获取2022年和2023年的描述性统计信息
stats_2022 = x_data_2022.describe()
stats_2023 = x_data_2023.describe()

# 计算两个年份数据的均值、方差差异
mean_diff = stats_2023.loc['mean'] - stats_2022.loc['mean']
std_diff = stats_2023.loc['std'] - stats_2022.loc['std']

# 显示均值和方差的差异
print("Mean Difference:")
print(mean_diff)

print("\nStandard Deviation Difference:")
print(std_diff)

stats_2022_y = y_data_2022[['DC_Voltage', 'DC_Current']].describe()
stats_2023_y = y_data_2023[['DC_Voltage', 'DC_Current']].describe()

# 计算两个年份数据的均值、方差差异
mean_diff_y = stats_2023_y.loc['mean'] - stats_2022_y.loc['mean']
std_diff_y = stats_2023_y.loc['std'] - stats_2022_y.loc['std']

# 显示均值和方差的差异
print("Mean Difference for y_data (DC_Voltage, DC_Current):")
print(mean_diff_y)

print("\nStandard Deviation Difference for y_data (DC_Voltage, DC_Current):")
print(std_diff_y)
