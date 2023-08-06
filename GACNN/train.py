import argparse
from utils.train_utils import train_utils


def parse_args():
    parser = argparse.ArgumentParser()
    # basic parameters
    # ===================================================dataset parameters=============================================================================
    parser.add_argument('--dataset_name', type=str, default='SEU',help='the name of the dataset：CWRU、SEU、MFPT')

    parser.add_argument('--dataset_path', type=str,default=r"D:\Users\hcy\Desktop\故障诊断数据集\SEU\Mechanical-datasets-master\gearbox\bearingset",help='the file path of the dataset')
    #parser.add_argument('--dataset_path', type=str, default=r"D:\Users\hcy\Desktop\故障诊断数据集\MFPT Fault Data Sets",help='the file path of the dataset')
    #parser.add_argument('--dataset_path', type=str, default=r"D:\Users\hcy\PycharmProjects\GNN_FD-main\图神经网络故障诊断\cwru", help='the file path of the dataset')

    parser.add_argument('--dir_path', type=str, default='12DE',help='the sample frequency of CWRU：12DE、48DE represent 12kHZ and 48kHZ respectively')
    parser.add_argument('--SEU_channel', type=int, default=0, help='the channel number of SEU：0-7')
    # XJTU
    parser.add_argument('--minute_value', type=int, default=5,help='the last (minute_value) csv file of XJTU datasets each fault class')
    parser.add_argument('--XJTU_channel', type=str, default='X', help='XJTU channel signal:X 、Y 、XY')

    # ===================================================data preprocessing parameters=============================================================================
    parser.add_argument('--sample_num', type=int, default=50, help='CWRU the number of samples')
    #parser.add_argument('--sample_num', type=int, default=100, help='MFPT the number of samples')
    # parser.add_argument('--sample_num', type=int, default=100, help='SEU the number of samples')
    parser.add_argument('--train_size', type=float, default=0.6, help='train size')
    parser.add_argument('--sample_length', type=int, default=1024, help='the length of each samples')
    parser.add_argument('--overlap', type=int, default=1024,help='the sampling shift of neibor two samples')  # 滑窗采样偏移量，当sample_length = overlap时为无重叠顺序采样
    parser.add_argument('--norm_type', type=str, default='Max-Min Normalization', help='the normlized method')  # 归一化方式
    parser.add_argument('--noise', type=int, default=0, help='whether add noise')  # 是否添加噪音
    parser.add_argument('--snr', type=int, default=-3, help='the snr of noise')

    parser.add_argument('--input_type', type=str, default='FD',help='TD——time domain signal，FD——frequency domain signal')  # 输入信号： 时域、频域
    parser.add_argument('--graph_type', type=str, default='complete_graph', help='the type of graph')  # 图类型
    parser.add_argument('--knn_K', type=int, default=4, help='the K value of knn-graph')
    parser.add_argument('--ER_p', type=float, default=0.5, help='the p value of ER-graph')
    parser.add_argument('--BA_n', type=int, default=4, help='the p value of BA-graph')
    parser.add_argument('--node_num', type=int, default=10, help='the number of node in a graph')  # 节点个数
    parser.add_argument('--direction', type=str, default='undirected', help='directed、undirected')  # 图的有向性
    parser.add_argument('--edge_type', type=str, default='Gaussian kernel',help='the edge weight method of graph')  # 边加权方式
    parser.add_argument('--edge_norm', type=bool, default=True, help='whether normalize edge weight')  # 是否将加权边归一化
    parser.add_argument('--batch_size', type=int, default=32)  # 批次大小

    # ========================= ==========================model =============================================================================
    parser.add_argument('--model_type', type=str, default='GACNN_2',help='the model of training and testing')  # 网络模型：GCN
    parser.add_argument('--epochs', type=int, default=100)  # 迭代轮数
    parser.add_argument('--learning_rate', type=float, default=0.001)  # 学习率
    parser.add_argument('--momentum', type=float, default=0.9)  # 动量因子
    parser.add_argument('--optimizer', type=str, default='Adam')  # 优化器

    # ===================================================visualization parameters=============================================================================
    parser.add_argument('--visualization', type=bool, default=False,help='whether visualize')  # 是否绘制混淆矩阵与每一层网络的t-SNE可视化图

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train_utils(args)

