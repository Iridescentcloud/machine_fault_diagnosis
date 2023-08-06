import numpy as np
from matplotlib import pyplot as plt, font_manager

def plot_embedding(data, label,classes):  # data -- 降维后的数据   label -- 标签   classes -- 类别  10
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(4,4))
    #fig = plt.figure()

    plt.rcParams['xtick.direction'] = 'in'  # 将x轴的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内`

    ax = plt.subplot(111)
    # 创建局部变量存放Plot以绘制相应的legend
    fig_leg = []
    list(map(lambda x: fig_leg.append(x), range(classes))) # [0,1,2,3,4,5,6,7,8,9]
    #marker = ['o', '^', 'p', 'P', '*', 's', 'x', 'X', '+', 'd', 'D', '>', 'H', 'h', '<', '1', '2']
    marker = ['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o']
    #marker = ['.','.','.','.','.','.','.','.','.','.','.','.','.','.','.']
# marker=marker[int(label[i])],
    # markersize=9,
    for i in range(data.shape[0]):  # 行
        fig_leg[int(label[i])] = plt.plot(data[i, 0], data[i, 1], linestyle='',marker=marker[int(label[i])],
                                          markersize=5, color=plt.cm.tab20(label[i] / 20.))
    my_font = font_manager.FontProperties(family='Times New Roman',size=7)  # 字体参数

    hand = list(map(lambda x: x[0], fig_leg))
# bbox_to_anchor=(1.12, 0.5)
    plt.legend(loc='right', ncol=1, frameon=True, labelspacing=0.8, columnspacing=0.4, handletextpad=0.4,
               prop=my_font, handlelength=1.5, bbox_to_anchor=(1.14, 0.5),
               handles=hand, labels=list(range(classes)))
    # ncol:图例列数   frameon:是否显示图列边框  labelspacing:图列中标签之间的垂直距离  handlelength:图例句柄条形的长度
    # columspacing: 调整图例中不同列之间的间距  label:图例中的名称    handletextpad:调节图例和标签文字之间的距离
    # facecolor：设置图例的背景颜色   prop:字体参数

    # plt.xticks(fontproperties=font_manager.FontProperties(family='Times New Roman',size=8))
    # plt.yticks(fontproperties=font_manager.FontProperties(family='Times New Roman',size=8))
    plt.xticks([])
    plt.yticks([])
    plt.xlim([data[:,0].min()-0.05,data[:,0].max()+0.05])
    plt.ylim([data[:,1].min()-0.05,data[:,1].max()+0.05])

    return fig