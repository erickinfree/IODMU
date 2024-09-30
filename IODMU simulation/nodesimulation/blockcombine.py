import numpy as np
import networkx as nx
import random
import copy

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

if __name__ == '__main__':
    layer11 = np.load('layer11.npy')
    layer22 = np.load('layer22.npy')
    layer33 = np.load('layer33.npy')
    print(len(layer11))
    print(len(layer22))
    print(len(layer33))
    seq1 = [ele.shape for ele in layer11]
    seq2 = [ele.shape for ele in layer22]
    seq3 = [ele.shape for ele in layer33]
    print(seq1)
    print(seq2)
    print(seq3)
    layer1_start = layer11[0][:20,:20]
    layer2_start = layer22[0][:30,:30]
    layer3_start = layer33[0][:40,:40]

    print(layer1_start)
    print(layer2_start)
    print(layer3_start)

    length = len(layer33)
    layer_11 = [layer1_start] * length
    layer_22 = [layer2_start] * length
    layer_33 = layer33

    layer_13 = np.load('layer13.npy')
    layer_31 = np.load('layer31.npy')
    layer_23 = np.load('layer23.npy')
    layer_32 = np.load('layer32.npy')
    layer_12 = [generate_binary_matrix(20,30)] * length
    layer_21 = [generate_binary_matrix(30,20)] * length

    tagmat_12 = np.random.rand(20, 30)
    layer_12 = [np.multiply(mat, tagmat_12) for mat in layer_12]

    tagmat_21 = np.random.rand(30, 20)
    layer_21 = [np.multiply(mat, tagmat_21) for mat in layer_21]

    tagmat_13 = np.random.rand(20, 80)
    layer_13 = [np.multiply(mat, tagmat_13) for mat in layer_13]

    tagmat_31 = np.random.rand(80, 20)
    layer_31 = [np.multiply(mat, tagmat_31) for mat in layer_31]

    tagmat_23 = np.random.rand(30, 80)
    layer_23 = [np.multiply(mat, tagmat_23) for mat in layer_23]

    tagmat_32 = np.random.rand(80, 30)
    layer_32 = [np.multiply(mat, tagmat_32) for mat in layer_32]


    block_matrix_seq = [np.array([[layer_11[i], layer_12[i], layer_13[i]],
                                  [layer_21[i], layer_22[i], layer_23[i]],
                                  [layer_31[i], layer_32[i], layer_33[i]]])
                        for i in range(length)]

    print(block_matrix_seq)

    a = np.array([[layer_11[0].shape, layer_12[0].shape, layer_13[0].shape],
                  [layer_21[0].shape, layer_22[0].shape, layer_23[0].shape],
                  [layer_31[0].shape, layer_32[0].shape, layer_33[0].shape]])

    print(a)

    np.save('block_matrix_seq33.npy', block_matrix_seq)