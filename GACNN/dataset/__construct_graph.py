import numpy as np
import torch
import networkx as nx
import scipy.spatial.distance as dist   # 距离计算模块distance

# Data 使用邻接表去表示图，同时也表示了node节点特征x, 边属性edge_attr(e.g 权重，类型)  Data只表示一张图
from torch_geometric.data import Data
from torch_cluster import knn_graph

'''
path_graph    路图
Knn_graph     Knn图
complete_graph  全连接图
ER_graph      ER随即图
'''

def edge_weight(x,edge_index,distance_type,edge_norm):  #边加权方式
    '''
    :param x: 每个图的节点特征  torch.tensor(10,1024)
    :param edge_index: 连边信息
    :param distace_type: 边加权度量方式
    :return:
    '''
    if distance_type == '0-1':
        edge_attr = np.ones(edge_index.shape[1])
        return edge_attr
    else:
        edge_index, x = np.array(edge_index), np.array(x)
        edge_attr = np.empty(edge_index.shape[1])     # path_graph : edge_index.shape[1] = 18 条边
        for edge_num in range(edge_index.shape[1]):  # 0~18
            source_node, target_node = edge_index[0][edge_num], edge_index[1][edge_num]  #取出源节点与目标节点编号
            source_node_feature, target_node_feature = x[source_node], x[target_node]   #取出源节点与目标节点特征
            if distance_type == 'Euclidean Distance':   #欧几里得距离
                distance = np.sqrt(np.sum(np.square(source_node_feature - target_node_feature)))
            elif distance_type == 'Manhattan Distance':  # 曼哈顿距离
                distance = np.sum(np.abs(source_node_feature - target_node_feature))
            elif distance_type == 'Chebyshev Distance':  # 切比雪夫距离
                distance =  np.abs(source_node_feature - target_node_feature).max()
            elif distance_type == 'Minkowski Distance':  # 闵可夫斯基距离
                distance = np.linalg.norm(source_node_feature - target_node_feature, ord=1)  #ord范数
            elif distance_type == 'Hamming Distance':  # 汉明距离
                distance = np.shape(np.nonzero(source_node_feature - target_node_feature)[0])[0]
            elif distance_type == 'Cosine Distance':  # 余弦相似度
                distance = np.dot(source_node_feature,target_node_feature)/(np.linalg.norm(source_node_feature)*(np.linalg.norm(target_node_feature)))
            elif distance_type == 'Pearson Correlation Coefficient':  # 皮尔逊相关系数
                distance = np.corrcoef(source_node_feature, target_node_feature)  #皮尔森相关系数矩阵
                distance = np.abs(distance[0,1])
            elif distance_type == 'Jaccard Similarity Coefficient':  #杰卡德相似系数,1
                distance = dist.pdist(np.array([source_node_feature, target_node_feature]), "jaccard")
            elif distance_type == 'Gaussian kernel':  # 高斯核权值函数
                beta = 0.01
                Euclidean_distance = np.sqrt(np.sum(np.square(source_node_feature - target_node_feature)))
                distance = np.exp(-Euclidean_distance/(2*beta**beta))

            edge_attr[edge_num] = distance

        if edge_norm == True:   # 边权重归一化
            edge_attr = (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min())  #归一化
        return edge_attr   # 边权重

#路图
def path_graph(data,direction,edge_type,label,edge_norm):
    '''
    每两个相邻样本之间有一条边
    :param data: 每个图的节点特征  (节点数，节点特征数) ----(10,1024)
    :param direction: 有向图、无向图
    :param edge_type: 边加权方式
    :param label: 节点标签
    :param edge_norm: 边权重是否归一化
    :return: path graph
    '''
    x = torch.tensor(data,dtype=torch.float) #节点特征
    if direction == 'directed':  #有向图
        edge_index = torch.tensor(np.array(nx.path_graph(data.shape[0]).edges).T, dtype=torch.long)
# edge_index  ([[0, 1, 2, 3, 4, 5, 6, 7, 8],
    #           [1, 2, 3, 4, 5, 6, 7, 8, 9]])
    elif direction == 'undirected':  # 无向图
        edge_index = torch.tensor(np.concatenate((np.array(nx.path_graph(data.shape[0]).edges).T,
                        np.roll(np.array(nx.path_graph(data.shape[0]).edges).T, shift=1, axis=0)),axis=1), dtype=torch.long)
#axis=1  np.concatenate 按行拼接
    edge_attr = edge_weight(x=x,edge_index=edge_index,distance_type=edge_type,edge_norm=edge_norm)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  #边权重
    y = torch.tensor(label * np.ones(data.shape[0]), dtype=torch.long)  # 节点标签

    graph = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)  #图

    return graph

#K-近邻图 -- 有向图
# 在KNNGraph中，可以找到每个节点的前k个最近邻
def Knn_graph(data,edge_type,label,edge_norm,K):
    '''
    :param data: 每个图的节点特征
    :param edge_type: 边加权方式
    :param label: 节点标签
    :param edge_norm: 边权重是否归一化
    :param K：邻居数
    :return: knn graph
    '''
    data = (data[:, :, 1] if data.shape[-1] == 2 else data)  # 若取时域+频域信号，则计算频域信号的皮尔森相关系数进行GNN
    x = torch.tensor(data, dtype=torch.float)  # 节点特征
    batch = torch.tensor(np.repeat(0,data.shape[0]), dtype=torch.int64)
    edge_index = knn_graph(x, k=K, batch=batch, loop=False, flow='target_to_source')  # K为邻居数，此处为有向图
    edge_attr = edge_weight(x=x, edge_index=edge_index, distance_type=edge_type, edge_norm=edge_norm)  # 边加权方式
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    y = torch.tensor(label * np.ones(data.shape[0]), dtype=torch.long)  # 节点标签

    graph = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)  # 图

    return graph

#全连接图
def complete_graph(data,edge_type,label,edge_norm):
    '''
    :param data: 每个图的节点特征
    :param edge_type: 边加权方式
    :param label: 节点标签
    :param edge_norm: 边权重是否归一化
    :return: complete graph
    '''
    x = torch.tensor(data, dtype=torch.float)  # 节点特征
    edge_index = []
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous()

    edge_attr = edge_weight(x=x, edge_index=edge_index, distance_type=edge_type, edge_norm=edge_norm)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # 边权重
    y = torch.tensor(label * np.ones(data.shape[0]), dtype=torch.long)  # 节点标签

    graph = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)  # 图

    return graph

#ER随机图--无向图
def ER_graph(data,edge_type,label,edge_norm,p):
    '''
    :param data: 每个图的节点特征
    :param edge_type: 边加权方式
    :param label: 节点标签
    :param edge_norm: 边权重是否归一化
    :param p: 任意两节点连接的概率值
    :return: ER random graph
    '''
    x = torch.tensor(data, dtype=torch.float)  # 节点特征
    edge_index = np.array(nx.random_graphs.erdos_renyi_graph(data.shape[0],p).edges).T  # random_graphs.erdos_renyi_graph(n,p)方法生成一个含有n个节点、以概率p连接的ER随机图
    edge_index = np.concatenate((edge_index, np.roll(edge_index, shift=1, axis=0)),axis=1)  # 有向图
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    edge_attr = edge_weight(x=x, edge_index=edge_index, distance_type=edge_type, edge_norm=edge_norm)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # 边权重
    y = torch.tensor(label * np.ones(data.shape[0]), dtype=torch.long)  # 节点标签

    graph = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)  # 图

    return graph

def WS_graph(data,edge_type,label,edge_norm,K,p):

    x = torch.tensor(data, dtype=torch.float)
    edge_index = np.array(nx.random_graphs.watts_strogatz_graph(data.shape[0],K,p).edges).T
    edge_index = np.concatenate((edge_index, np.roll(edge_index, shift=1, axis=0)), axis=1)  # 有向图
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    edge_attr = edge_weight(x=x, edge_index=edge_index, distance_type=edge_type, edge_norm=edge_norm)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # 边权重
    y = torch.tensor(label * np.ones(data.shape[0]), dtype=torch.long)  # 节点标签

    graph = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)  # 图

    return graph

def BA_graph(data,edge_type,label,edge_norm,n):

    x = torch.tensor(data, dtype=torch.float)
    edge_index = np.array(nx.random_graphs.barabasi_albert_graph(data.shape[0],n).edges).T
    edge_index = np.concatenate((edge_index, np.roll(edge_index, shift=1, axis=0)), axis=1)  # 有向图
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    edge_attr = edge_weight(x=x, edge_index=edge_index, distance_type=edge_type, edge_norm=edge_norm)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)  # 边权重
    y = torch.tensor(label * np.ones(data.shape[0]), dtype=torch.long)  # 节点标签

    graph = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)  # 图

    return graph


def generate_graph(feature,graph_type,node_num,direction,edge_type,edge_norm,K,p,n):
    '''
    :param feature: shape (classes，sample_num，sample_length) (10,200,1024) classes-故障类型数；sample_num-每种故障样本数；sample_length-每个样本长度
    :param graph_type: 图类型
    :param node_num: 每个图的节点个数
    :param direction: 有向图、无向图
    :param edge_type: 边加权方式
    :param edge_norm: 边权重归一化
    :param K: knn graph的邻居数
    :param p: ER random graph的任意两节点的概率
    :return graph_dataset: 图数据集 -- 列表(故障类型数，图个数)
    '''

    graph_dataset = []  # 按照故障类型逐次将图数据存入空列表中 图数据集
# feature : data  shape (10，200，1024)
    for label, class_fea in enumerate(feature):  #  class_fea 每种故障类型的特征值 (200,1024)
        # np.random.shuffle(class_fea)  #将取出来的每一类故障信号shuffle
        start = 0
        end = node_num  # 10

        while end <= class_fea.shape[0]:  # end < 行数
            a_graph_fea = class_fea[start:end,:]  # 每一个图的节点特征  前10行

# a_graph_fea  (10,1024)
            if graph_type == 'path_graph':
                graph = path_graph(data=a_graph_fea,direction=direction,edge_type=edge_type,label=label,edge_norm=edge_norm)
            elif graph_type == 'knn_graph':
                graph = Knn_graph(data=a_graph_fea,edge_type=edge_type,label=label,edge_norm=edge_norm,K=K)
            elif graph_type == 'complete_graph':
                graph = complete_graph(data=a_graph_fea,edge_type=edge_type,label=label,edge_norm=edge_norm)
            elif graph_type == 'ER_graph':
                graph = ER_graph(data=a_graph_fea,edge_type=edge_type,label=label,edge_norm=edge_norm,p=p)
            elif graph_type == 'WS_graph':
                graph = WS_graph(data=a_graph_fea, edge_type=edge_type, label=label, edge_norm=edge_norm,K=K, p=p)
            elif graph_type == 'BA_graph':
                graph = BA_graph(data=a_graph_fea, edge_type=edge_type, label=label, edge_norm=edge_norm,n=n)
            else:
                print('this graph is not existed!!!')

            start = start + node_num
            end = end + node_num

            graph_dataset.append(graph)

    return graph_dataset


