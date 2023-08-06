import numpy as np
import os
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from dataset._user_functions import Normal_signal,Slide_window_sampling,Add_noise,FFT
from dataset.__construct_graph import generate_graph

def data_preprocessing(dataset_path,sample_number,window_size,overlap,normalization,noise,snr,
                         input_type,graph_type,K,p,n,node_num,direction,edge_type,edge_norm,train_size,batch_size):

    root = dataset_path

    dir = ['1 - Three Baseline Conditions', '3 - Seven More Outer Race Fault Conditions',
           '4 - Seven Inner Race Fault Conditions']
    mat_name = [['baseline_1.mat'],
                ['OuterRaceFault_vload_1.mat', 'OuterRaceFault_vload_2.mat', 'OuterRaceFault_vload_3.mat',
                 'OuterRaceFault_vload_4.mat',
                 'OuterRaceFault_vload_5.mat', 'OuterRaceFault_vload_6.mat', 'OuterRaceFault_vload_7.mat'],
                ['InnerRaceFault_vload_1.mat', 'InnerRaceFault_vload_2.mat', 'InnerRaceFault_vload_3.mat',
                 'InnerRaceFault_vload_4.mat', 'InnerRaceFault_vload_5.mat', 'InnerRaceFault_vload_6.mat',
                 'InnerRaceFault_vload_7.mat']]

    data = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    data_index = 0
    for num, each_dir in enumerate(dir):
        for each_mat in mat_name[num]:
            f = loadmat(os.path.join(root, each_dir, each_mat))
            if num == 0:  # num=0时为正常信号
                data[data_index].append(f['bearing'][0][0][1].squeeze(axis=1)[:146484])  # 取正常样本前146484，使得与其余故障样本数平衡
            else:
                data[data_index].append(f['bearing'][0][0][2].squeeze(axis=1))

            data_index = data_index + 1

    data = np.array(data).squeeze(axis=1)  # shape:(15,146484)

    # 添加噪声
    if noise == 1 or noise == 'y':
        noise_data = np.zeros((data.shape[0], data.shape[1]))
        for data_i in range(data.shape[0]):
            noise_data[data_i] = Add_noise(data[data_i], snr)
    else:
        noise_data = data

    # 滑窗采样
    sample_data = np.zeros((noise_data.shape[0], noise_data.shape[1] // window_size, window_size))
    for noise_data_i in range(noise_data.shape[0]):
        sample_data[noise_data_i] = Slide_window_sampling(noise_data[noise_data_i], window_size=window_size,
                                                          overlap=overlap)

    sample_data = sample_data[:, :sample_number, :]
    # 归一化
    if normalization != 'unnormalization':
        norm_data = np.zeros((sample_data.shape[0], sample_data.shape[1], sample_data.shape[2]))
        for sample_data_i in range(sample_data.shape[0]):
            norm_data[sample_data_i] = Normal_signal(sample_data[sample_data_i], normalization)
    else:
        norm_data = sample_data

    if input_type == 'TD':  #时域信号
        data = norm_data
    elif input_type == 'FD':  #频域信号
        data = np.zeros((norm_data.shape[0],norm_data.shape[1],norm_data.shape[2]))
        for label_index in range(norm_data.shape[0]):
            fft_data = FFT(norm_data[label_index,:,:])
            data[label_index,:,:] = fft_data

    graph_dataset = generate_graph(feature=data, graph_type=graph_type, node_num=node_num, direction=direction,
                                   edge_type=edge_type, edge_norm=edge_norm, K=K, p=p,n=n)

    str_y_1 = []
    for i in range(len(graph_dataset)):
        str_y_1.append(np.array(graph_dataset[i].y))


    train_data, test_data = train_test_split(graph_dataset, train_size=train_size, shuffle=True,random_state=1,stratify=str_y_1)  # 训练集、测试集划分

    loader_train = DataLoader(train_data,batch_size=batch_size)
    loader_test = DataLoader(test_data,batch_size=batch_size)

    return loader_train, loader_test

loader_train, loader_test = data_preprocessing(dataset_path=r"D:\Users\hcy\Desktop\故障诊断数据集\MFPT Fault Data Sets",
                             sample_number=100,window_size=1024,overlap=1024,normalization='Max-Min Normalization',
                             noise=0,snr=0,input_type='FD',graph_type='complete_graph',K=4,p=0.5, n=4,
                             node_num=10,direction='undirected',
                             edge_type='Gaussian kernel',edge_norm=True,batch_size=64,train_size=0.6)