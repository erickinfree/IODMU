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

def generate_binary_matrix(rows, cols):
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
                matrix[i, j] = np.random.randint(2)


        # 确保每行和每列至少有一个1
        if np.all(np.any(matrix == 1, axis=1)) and np.all(np.any(matrix == 1, axis=0)):
            return matrix

# 生成边数递增的矩阵序列
def edgeRise_matrix_seq(startmat, num_ones):
    matrix_seq = []
    current_num_ones = np.sum(startmat)
    print(current_num_ones)
    startmat = startmat
    while current_num_ones < num_ones:
        # Make a copy of the startmat to avoid modifying it in place
        newmat = startmat.copy()

        # Randomly select a position and set it to 1 if it's 0
        row, col = np.random.randint(newmat.shape[0]), np.random.randint(newmat.shape[1])
        if newmat[row, col] == 0:
            newmat[row, col] = 1
        else:
            while newmat[row, col] != 0:
                row, col = np.random.randint(newmat.shape[0]), np.random.randint(newmat.shape[1])
            newmat[row, col] = 1


        matrix_seq.append(newmat)
        current_num_ones = np.sum(newmat)
        startmat = newmat
        print(newmat)
        print(current_num_ones)

    return matrix_seq



# 生成一个5x5的0和1随机矩阵，对角线为0
rows, cols = 30, 40
binary_matrix = generate_binary_matrix(rows, cols)
print(binary_matrix)

res = edgeRise_matrix_seq(binary_matrix,1199)


tagmat = np.random.rand(rows, cols)

matseq = [np.multiply(mat,tagmat) for mat in res]

print(matseq)
print(len(matseq))
np.save('layer23edgeseq.npy',matseq )