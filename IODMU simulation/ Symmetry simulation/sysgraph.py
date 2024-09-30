import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(25, 4))
sysseq = np.load('sysseq.npy',allow_pickle=True)
print(sysseq)



# # 示例数据
x = list(range(len(sysseq[0])))
print(x)
y1 = [190-i for i in range(len(sysseq[0]))]
print(y1)
# y2 = [ele[1][2] for ele in layer33]
# print(len(y1))
# print(len(y2))
#
# 创建图形和轴对象
fig, ax1 = plt.subplots()

# 绘制第一个数据集和坐标轴（左侧）
color1 = 'blue'
ax1.plot(x, y1, color=color1,
         marker='o',
         label='Number of Edges',
         markersize=1,
         zorder=101)
ax1.set_xlabel('Period of Change')
ax1.set_ylabel('Number of Edges', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
lines, labels = ax1.get_legend_handles_labels()
plt.ylim(0, 240)
#
# 创建第二个坐标轴（右侧）
ax2 = ax1.twinx()
color2 = '#FF8000'
for line in sysseq[:-1]:
    ax2.plot(x, line, color=color2,
             marker='x',
             markersize=1,
             zorder=10)
ax2.plot(x, sysseq[-1], color=color2,
         marker='x',
         label='IO_modified',
         markersize=1,
         zorder=13)
ax2.set_ylabel('Change in Inversion Number', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
plt.ylim(0, 150)

# 重新激活 ax1 以确保其在 ax2 之上
ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)  # 隐藏 ax1 的背景，使其透明，以便能看到 ax2 的线条


# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

# #
#
# # 显示图形
plt.savefig('IOsys.svg', format='svg')
plt.show()
