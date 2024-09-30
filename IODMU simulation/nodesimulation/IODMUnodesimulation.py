import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

# 增加一个节点函数
def nodePlus(matrix):
    dim = len(matrix)
    new_column = np.random.randint(0, 2, size=(1, dim))
    new_column = new_column[0]
    while True:
        if np.sum(new_column) != 0:
            break
        else:
            new_column = np.random.randint(0, 2, size=(1, dim))

    new_matrix = np.hstack((matrix, new_column[:, np.newaxis]))
    new_row = np.append(new_column,0)
    new_matrix = np.vstack((new_matrix,new_row))
    return new_matrix

# matrix为初始矩阵，num代表增加节点数的步数
def nodeIncreaseSeq(matrix,num):
    startnode = len(matrix)
    startedge = 0.5 * np.sum(matrix)
    dim = len(matrix) + num
    startmat = matrix
    startmatcol = np.zeros((len(startmat), dim - len(startmat)))
    startmatrow = np.zeros((dim - len(startmat), dim))
    res_step1 = np.hstack((startmat, startmatcol))
    res_step2 = np.vstack((res_step1, startmatrow))

    mat_seq = [res_step2]
    node_seq = [startnode]
    edge_seq = [startedge]

    dic = {}
    i = 1
    while i <= num:
        endmat = nodePlus(startmat)
        matcol = np.zeros((len(endmat),dim-len(endmat)))
        matrow = np.zeros((dim-len(endmat),dim))
        res1 = np.hstack((endmat,matcol))
        res2 = np.vstack((res1,matrow))
        mat_seq.append(res2)
        node_seq.append(len(endmat))
        edge_seq.append(0.5 * np.sum(endmat))
        startmat = endmat
        i += 1

    dic['seq'] = mat_seq
    dic['nodes'] = node_seq
    dic['edges'] = edge_seq
    return dic

if __name__ == '__main__':
    rows, cols = 40, 40
    startmat = generate_binary_matrix_with_zero_diagonal(rows, cols)
    nodseq = nodeIncreaseSeq(startmat, 40)
    print(nodseq)
    tagmat = np.random.rand(nodseq['seq'][0].shape[0], nodseq['seq'][0].shape[1])
    print(tagmat)
    matseq = [np.multiply(mat, tagmat) for mat in nodseq['seq']]
    print(matseq)
    np.save('layer33.npy',matseq)