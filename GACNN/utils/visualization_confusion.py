import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.confusion import confusion

def visualization_confusion(loader_test,prediction):  # 测试集 、 预测标签
    import warnings
    warnings.filterwarnings('ignore')
    import numpy as np
    from sklearn.manifold import TSNE
    from utils.tsne import plot_embedding

    def visualization_tsne(loader_test, y_feature, classes):  # 参数： 测试集loader 、 特征值 、 classes=output_dim 类别数

        # the label of testing dataset  测试集的标签
        label = np.empty(0, )  # 空列表
        for i in range(len(loader_test.dataset)):  # len(loader_test.dataset) ---- 测试集长度  eg:120
            label = np.append(label, loader_test.dataset[i].y)  # 把测试的标签依次添加到 label 列表中   len(loader_test.dataset)*10

        # tsne
        warnings.filterwarnings('ignore')  # 警告过滤器  'ignore' ---忽略匹配的警告
        tsne = TSNE(n_components=2, init='pca')  # 初始化一个tsne实例   2 维

        result = tsne.fit_transform(y_feature)  # 对特征进行降维
        fig = plot_embedding(result, label, classes=classes)

    #the label of testing dataset
    label = np.empty(0,)
    for i in range(len(loader_test.dataset)):
        label = np.append(label,loader_test.dataset[i].y)  # 把测试机标签添加到 label 列表中

    #confusion matrix   混淆矩阵
    confusion_data = confusion_matrix(label,prediction)  # label---真实标签   prediction---预测标签

    confusion(confusion_matrix=confusion_data)  #混淆矩阵绘制



