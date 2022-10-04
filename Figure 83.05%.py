import numpy as np
import matplotlib.pyplot as plt
import xlrd
from matplotlib.ticker import MultipleLocator

workbook = xlrd.open_workbook(r'C:\Users\戴尔\Desktop\Staggered contrast(83.05%).xls')  # window下要加r
sheet = workbook.sheet_by_index(0)
loss1 = []
acc1 = []
loss2 = []
acc2 = []
for i in range(1, sheet.nrows):
    loss1.append(sheet.cell_value(i, 0))
    acc1.append(sheet.cell_value(i, 1))
    loss2.append(sheet.cell_value(i, 2))
    acc2.append(sheet.cell_value(i, 3))

# print(loss1)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.tight_layout()  # 解决绘图时上下标题重叠现象
plt.rcParams.update({'font.size': 12})  # 设置图例字体大小

plt.figure(figsize=(4, 3))
plt.plot(loss1, label='Training Loss',  linewidth=3.0,)  # 加粗
plt.plot(acc1, label='Training Accuracy', linewidth=3.0, )
plt.plot(loss2, label='Validation Loss', linewidth=3.0, )
plt.plot(acc2, label='Validation Accuracy', linewidth=3.0, )

plt.title('Staggered contrast(83.05%)', fontsize=16, fontweight='bold')
plt.xlabel(xlabel='Epoch', fontsize=12, fontweight='bold')
plt.ylabel(ylabel='Value', fontsize=12, fontweight='bold')
plt.ylim(0, 2)
plt.xlim(0, 100)
plt.tick_params(labelsize=11)
y_major_locator = MultipleLocator(10)
# 把y轴的刻度间隔设置为10，并存在变量里
# ax = plt.gca()
# ax.yaxis.set_major_locator(y_major_locator)
# # 把y轴的主刻度设置为10的倍数
plt.legend()

# plt.savefig(r'E:\model original\picture_1.tiff')
plt.show()
