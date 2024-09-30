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
# 层间超边扩张函数

def supermatPluse(matrix,dim_row,dim_col):
    rownum = matrix.shape[0]
    colnum = matrix.shape[1]
    if rownum < dim_row and colnum < dim_col:
        matup = np.zeros((rownum, dim_col - colnum))
        matdown = np.zeros((dim_row - rownum, dim_col))
        matrix_new = np.hstack((matrix,matup))
        matrix_new = np.vstack((matrix_new,matdown))
        return matrix_new
    elif rownum < dim_row and colnum == dim_col:
        matdown = np.zeros((dim_row - rownum, dim_col))
        matrix_new = np.vstack((matrix, matdown))
        return matrix_new
    elif rownum == dim_row and colnum < dim_col:
        matup = np.zeros((rownum, dim_col - colnum))
        matrix_new = np.hstack((matrix, matup))
        return matrix_new
    elif rownum == dim_row and colnum == dim_col:
        return matrix

def supermatSeqIncrease(mat_seq,dim_row,dim_col):
    new_matseq = [supermatPluse(ele,dim_row,dim_col) for ele in mat_seq]
    return new_matseq

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



    rows,cols = 40,30
    netseq1_2 = []
    for i in range(len(layer33)):
        rownum = rows+i
        colnum = cols
        midmat = np.random.randint(2, size=(rownum, colnum))
        netseq1_2.append(copy.copy(midmat))

    print(netseq1_2)
    print([ele.shape for ele in netseq1_2])
    mat12seq = supermatSeqIncrease(netseq1_2,80,30)
    print(mat12seq)
    print([ele.shape for ele in mat12seq])
    np.save('layer32.npy',mat12seq)


