import pandas as pd
import networkx as nx
import numpy as np

def generate_binary_matrix_with_zero_diagonal(rows, cols):
    """
    生成一个0和1的随机矩阵，对角线为0，并确保不存在全零行和全零列
    :param rows: 行数
    :param cols: 列数
    :return: 0和1的随机矩阵
    """
    while True:
        # 初始化全零矩阵
        matrix = np.zeros((rows, cols), dtype=int)

        # 随机生成0和1填充非对角线元素
        for i in range(rows):
            for j in range(cols):
                if i != j:
                    matrix[i, j] = np.random.randint(2)

        # 确保每行和每列至少有一个1
        if np.all(np.any(matrix == 1, axis=1)) and np.all(np.any(matrix == 1, axis=0)):
            return matrix


layer_11 = np.load('layer1edgeseq.npy')
layer_22 = np.load('layer2edgeseq.npy')
layer_33 = np.load('layer3edgeseq.npy')
print(layer_33[0].shape)

# =============================边变化===================================
length = len(layer_33)
#
# layer_12 = generate_binary_matrix_with_zero_diagonal(20, 30)
# layer_12 = [layer_12] * length
# tagmat_12 = np.random.rand(20, 30)
# layer_12 = [np.multiply(mat, tagmat_12) for mat in layer_12]
# np.save('layer12.npy',layer_12)
#
# layer_13 = generate_binary_matrix_with_zero_diagonal(20, 40)
# layer_13 = [layer_13] * length
# tagmat_13 = np.random.rand(20, 40)
# layer_13 = [np.multiply(mat, tagmat_13) for mat in layer_13]
# np.save('layer13.npy',layer_13)
#
# layer_23 = generate_binary_matrix_with_zero_diagonal(30, 40)
# layer_23 = [layer_23] * length
# tagmat_23 = np.random.rand(30, 40)
# layer_23 = [np.multiply(mat, tagmat_23) for mat in layer_23]
# np.save('layer23.npy',layer_23)
#
# layer_21 = generate_binary_matrix_with_zero_diagonal(30, 20)
# layer_21 = [layer_21] * length
# tagmat_21 = np.random.rand(30, 20)
# layer_21 = [np.multiply(mat, tagmat_21) for mat in layer_21]
# np.save('layer21.npy',layer_21)
#
# layer_31 = generate_binary_matrix_with_zero_diagonal(40, 20)
# layer_31 = [layer_31] * length
# tagmat_31 = np.random.rand(40, 20)
# layer_31 = [np.multiply(mat, tagmat_31) for mat in layer_31]
# np.save('layer31.npy',layer_31)
#
# layer_32 = generate_binary_matrix_with_zero_diagonal(40, 30)
# layer_32 = [layer_32] * length
# tagmat_32 = np.random.rand(40, 30)
# layer_32 = [np.multiply(mat, tagmat_32) for mat in layer_32]
# np.save('layer32.npy',layer_32)
#
#
layer_11 = [layer_11[0]] * length
np.save('layer11.npy',layer_22)
#
layer_22 = [layer_22[0]] * length
np.save('layer22.npy',layer_33)

