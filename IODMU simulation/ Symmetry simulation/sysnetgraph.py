import numpy as np
import pandas as pd

# 生成一个20x20的随机矩阵
matrix = np.random.rand(20, 20)

# 将对角线元素设置为0
np.fill_diagonal(matrix, 0)

# # 将矩阵转换为DataFrame
# df = pd.DataFrame(matrix)
#
# # 将DataFrame保存为Excel文件
# df.to_excel("20x20_matrix_with_zeros_on_diagonal.xlsx", index=False, header=False)
#
# print("20x20 matrix with zeros on the diagonal has been saved to 20x20_matrix_with_zeros_on_diagonal.xlsx")
#
