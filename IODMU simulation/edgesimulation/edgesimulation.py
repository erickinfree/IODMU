import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

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

# 生成边数递增的矩阵序列
def edgeRise_matrix_seq(startmat,num_ones):
    matrix_seq = []
    current_num_ones = np.sum(startmat)
    print(current_num_ones)
    while np.sum(startmat) < num_ones:
        row = np.random.randint(len(startmat))
        myset = set(range(len(startmat)))
        myset.discard(row)
        col = random.choice(list(myset))
        print(row,col)
        if startmat[row, col] == 0:
            startmat[row, col] = 1
            startmat[col, row] = 1
        else:
            # 如果当前位置的元素是 1，则循环直到不为1
            while startmat[row, col] != 0:
                row = np.random.randint(len(startmat))
                myset = set(range(len(startmat)))
                myset.discard(row)
                col = random.choice(list(myset))
                if startmat[row, col] == 0:
                    startmat[row, col] = 1
                    startmat[col, row] = 1
                    break
        matrix_seq.append(startmat.copy())
        # print(np.sum(startmat))
    return matrix_seq


# 生成一个5x5的0和1随机矩阵，对角线为0
rows, cols = 30, 30
binary_matrix = generate_binary_matrix_with_zero_diagonal(rows, cols)
print(binary_matrix)

res = edgeRise_matrix_seq(binary_matrix,870)


tagmat = np.random.rand(rows, cols)

matseq = [np.multiply(mat,tagmat) for mat in res]

print(matseq)
np.save('layer2edgeseq.npy',matseq )