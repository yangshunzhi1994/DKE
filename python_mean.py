import numpy as np

# 替换这里的data列表为你的数据
data = [77.51, 77.26, 77.86, 77.64, 77.73]

# 计算平均值
mean = np.mean(data)

# 计算标准差
std_dev = np.std(data, ddof=0)  # ddof=0 用于计算总体标准差

print("平均值:", mean)
print("标准差:", std_dev)
