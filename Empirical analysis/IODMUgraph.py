import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 创建一个图形并设置大小
plt.figure(figsize=(17, 5))

layerchange = np.load('layerchange2.npy')
y1 = [ele[3][3] for ele in layerchange]
y1 = [0] + y1


date_index = pd.date_range('2022/12', '2023/12', freq='M')
print(date_index)

df_y1 = pd.DataFrame({'Value': y1}, index=date_index)
print(df_y1)
date_index_np = date_index.to_numpy()
# 绘制月度时序折线图

plt.plot(date_index_np,
         df_y1['Value'].to_numpy(),
         label='Risk Value in Stock Layer.(IO_modified[4][4])',
         marker='o',color = 'orange')

# 添加标题和标签
plt.xlabel('month',fontsize=15)
plt.ylabel('IO_modified Value',fontsize=15)

# 添加图例
plt.legend(fontsize=15)

# 自动调整日期格式
# plt.gcf().autofmt_xdate()
plt.xticks(rotation=0,fontsize=15)
plt.xticks(fontsize=15)
plt.xlim([date_index[0]-datetime.timedelta(days=31), date_index[-1]+datetime.timedelta(days=31)])

# 显示图形
plt.savefig('IO[3][3].svg', format='svg')
plt.show()

