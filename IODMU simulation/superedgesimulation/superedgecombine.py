import numpy as np

layer12seq = np.load('layer12edgeseq.npy')
layer13seq = np.load('layer13edgeseq.npy')
layer21seq = np.load('layer21edgeseq.npy')
layer23seq = np.load('layer23edgeseq.npy')
layer31seq = np.load('layer31edgeseq.npy')
layer32seq = np.load('layer32edgeseq.npy')

layer11seq = np.load('layer1edgeseq.npy')
layer22seq = np.load('layer2edgeseq.npy')
layer33seq = np.load('layer3edgeseq.npy')

# ==================================layer12===================================
length = len(layer32seq)
layer11 = [layer11seq[0]] * length
layer22 = [layer22seq[0]] * length
layer33 = [layer33seq[0]] * length

layer12 = [layer12seq[0]] * length
layer13 = [layer13seq[0]] * length
layer23 = [layer23seq[0]] * length
layer21 = [layer21seq[0]] * length
layer31 = [layer31seq[0]] * length
layer32 = layer32seq

block_matrix_seq = [np.array([[layer11[i], layer12[i], layer13[i]],
                              [layer21[i], layer22[i], layer23[i]],
                              [layer31[i], layer32[i], layer33[i]]])
                    for i in range(length)]

print(block_matrix_seq)

a = np.array([[layer11[0].shape, layer12[0].shape, layer13[0].shape],
              [layer21[0].shape, layer22[0].shape, layer23[0].shape],
              [layer31[0].shape, layer32[0].shape, layer33[0].shape]])

print(a)

np.save('block_matrix_seq32.npy',block_matrix_seq)

