import torch
import numpy as np
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import CWRU_data,MFPT_data,SEU_data,xjtu_data
from model.ChebyNet import ChebyNet
from model.GCN import GCN
from model.CNN import CNN_1D
from model.MRF_GCN import MRF_GCN
from model.GACNN import GACNN
from model.GACNN2 import GACNN2
from model.GACNN_2 import GACNN_2
from utils.visualization_confusion import visualization_confusion
from utils.visualization_tsne import visualization_tsne


def train_utils(args):
    #==============================================================1、训练集、测试集===================================================
    if args.dataset_name == 'CWRU':
        loader_train, loader_test =CWRU_data.data_preprocessing(dataset_path=args.dataset_path,sample_number=args.sample_num,dir_path=args.dir_path,
                             window_size=args.sample_length,overlap=args.overlap,normalization=args.norm_type,noise=args.noise,snr=args.snr,input_type=args.input_type,
                             graph_type=args.graph_type,K=args.knn_K,p=args.ER_p, n=args.BA_n,node_num=args.node_num,direction=args.direction,
                             edge_type=args.edge_type,edge_norm=args.edge_norm,batch_size=args.batch_size,train_size=args.train_size)
        output_dim = 10 #10分类

    elif args.dataset_name == 'SEU':
        loader_train, loader_test = SEU_data.data_preprocessing(dataset_path=args.dataset_path,channel=args.SEU_channel,sample_number=args.sample_num,window_size=args.sample_length, overlap=args.overlap,
                                                                   normalization=args.norm_type, noise=args.noise,snr=args.snr,input_type=args.input_type,graph_type=args.graph_type, K=args.knn_K,
                                                                   p=args.ER_p,n=args.BA_n,node_num=args.node_num, direction=args.direction,edge_type=args.edge_type,
                                                                   edge_norm=args.edge_norm,batch_size=args.batch_size,train_size=args.train_size)
        output_dim = 10  # 10分类

    elif args.dataset_name == 'XJTU':
        loader_train, loader_test = xjtu_data.data_preprocessing(dataset_path=args.dataset_path,channel=args.XJTU_channel,minute_value=args.minute_value, sample_number=args.sample_num,
                                                                window_size=args.sample_length, overlap=args.overlap,normalization=args.norm_type, noise=args.noise,snr=args.snr, input_type=args.input_type,
                                                                graph_type=args.graph_type, K=args.knn_K,p=args.ER_p,n=args.BA_n,node_num=args.node_num, direction=args.direction,
                                                                edge_type=args.edge_type,edge_norm=args.edge_norm, batch_size=args.batch_size,train_size=args.train_size)
        output_dim = 15  # 15分类

    elif args.dataset_name == 'JNU':
        loader_train, loader_test = JNU_data.data_preprocessing(dataset_path=args.dataset_path,sample_number=args.sample_num,window_size=args.sample_length,overlap=args.overlap,normalization=args.norm_type,
                                                                noise=args.noise,snr=args.snr, input_type=args.input_type,graph_type=args.graph_type, K=args.knn_K,p=args.ER_p,n=args.BA_n,
                                                                node_num=args.node_num,direction=args.direction,edge_type=args.edge_type,edge_norm=args.edge_norm,batch_size=args.batch_size,train_size=args.train_size)
        output_dim = 12  # 12分类

    elif args.dataset_name == 'MFPT':
        loader_train, loader_test = MFPT_data.data_preprocessing(dataset_path=args.dataset_path,sample_number=args.sample_num,window_size=args.sample_length, overlap=args.overlap,normalization=args.norm_type,
                                                                noise=args.noise, snr=args.snr,input_type=args.input_type, graph_type=args.graph_type,K=args.knn_K, p=args.ER_p,n=args.BA_n,
                                                                node_num=args.node_num, direction=args.direction,edge_type=args.edge_type, edge_norm=args.edge_norm,batch_size=args.batch_size, train_size=args.train_size)
        output_dim = 15  # 15分类

    elif args.dataset_name == 'UoC':
        loader_train, loader_test = UoC_data.data_preprocessing(dataset_path=args.dataset_path,sample_number=args.sample_num,window_size=args.sample_length, overlap=args.overlap,normalization=args.norm_type,
                                                                noise=args.noise, snr=args.snr,input_type=args.input_type, graph_type=args.graph_type,K=args.knn_K, p=args.ER_p,
                                                                node_num=args.node_num, direction=args.direction,edge_type=args.edge_type, edge_norm=args.edge_norm,batch_size=args.batch_size, train_size=args.train_size)
        output_dim = 9  # 9分类

    elif args.dataset_name == 'DC':
        loader_train, loader_test = DC_data.data_preprocessing(dataset_path=args.dataset_path,sample_number=args.sample_num,window_size=args.sample_length, overlap=args.overlap,normalization=args.norm_type,
                                                               noise=args.noise, snr=args.snr,input_type=args.input_type, graph_type=args.graph_type,K=args.knn_K, p=args.ER_p,
                                                               node_num=args.node_num, direction=args.direction,edge_type=args.edge_type, edge_norm=args.edge_norm,batch_size=args.batch_size, train_size=args.train_size)
        output_dim = 10  # 10分类

    else:
        print('this dataset is not existed!!!')  # 输出

    #==============================================================2、网络模型===================================================
    input_dim = loader_train.dataset[0].x.shape[1]  # 1024  x---(10，1024) shape[1]--列数
    print(input_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   #设备

    if args.model_type == 'GCN':
        model = GCN(input_dim, output_dim) .to(device) # output_dim =10 (十分类)
    elif args.model_type == 'ChebyNet':
        model = ChebyNet(input_dim, output_dim).to(device)
    elif args.model_type == 'MRF_GCN':
        model = MRF_GCN(input_dim, output_dim).to(device)
    elif args.model_type == 'CNN':
        model = CNN_1D(in_channel=1,out_channel=output_dim).to(device)
    elif args.model_type == 'GACNN':
        model = GACNN(input_dim,output_dim).to(device)
    elif args.model_type == 'GACNN2':
        model = GACNN2(input_dim,output_dim).to(device)
    elif args.model_type == 'GACNN_2':
        model = GACNN_2(input_dim,output_dim).to(device)
    else:
        print('this model is not existed!!!')

    # ==============================================================3、超参数===================================================
    epochs = args.epochs  # 迭代次数100
    lr = args.learning_rate  # 学习率0.0001

    #==================================优化器====================================
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif args.optimizer == 'Momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=args.momentum)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif args.optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr)
    elif args.optimizer == 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    else:
        print('this optimizer is not existed!!!')

    # ==============================================================4、训练===================================================
    all_train_loss = []    # 训练损失
    all_train_accuracy = []  # 训练精度
    train_time = []    # 训练时间

    for epoch in range(epochs):

        start = time.perf_counter() # 返回当前的计算机系统时间

        model.train()  # 模型训练
        correct_train = 0
        train_loss = 0
        for step, train_data in enumerate(loader_train):
            train_data = train_data.to(device)   # train_data 一个批次 64张图
            train_out = model(train_data)
            #print(train_out)
            #print(model.get_fea()[0].shape)
            #print(model.get_fea()[1].shape)
            loss = F.nll_loss(train_out, train_data.y)  # 批次平均损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = train_loss + loss.item()
            pre_train = torch.max(train_out.cpu(), dim=1)[1].data.numpy()
            correct_train = correct_train + (pre_train == train_data.y.cpu().data.numpy()).astype(int).sum()

        end = time.perf_counter()

        train_time.append(end-start)  #记录训练时间
        len(loader_train.dataset)  # 训练集的图个数
        len(loader_train.dataset) * loader_train.dataset[0].num_nodes   # 所有的节点数，每张图10个节点
        train_accuracy = correct_train / (len(loader_train.dataset) * loader_train.dataset[0].num_nodes)
        all_train_loss.append(train_loss)
        all_train_accuracy.append(train_accuracy)

        print('epoch：{} '
              '| train loss：{:.2f} '
              '| train accuracy：{}/{}({:.2f}) '
              '| train time：{}(s/epoch)'.format(
            epoch,train_loss,correct_train,len(loader_train.dataset) * loader_train.dataset[0].num_nodes,100*train_accuracy,end-start))


    # ==============================================================5、测试===================================================
    y_fea = []
    list(map(lambda x:y_fea.append([]),range(len(model.get_fea()))))  # y_fea = [] 根据可视化的层数来创建相应数量的空列表存放特征

    prediction = np.empty(0,)  #存放预测标签绘制混淆矩阵
    model.eval()  # 模型测试
    correct_test = 0
    for test_data in loader_test:
        test_data = test_data.to(device)
        test_out = model(test_data)
        pre_test = torch.max(test_out.cpu(),dim=1)[1].data.numpy()  # 预测值
        correct_test = correct_test + (pre_test == test_data.y.cpu().data.numpy()).astype(int).sum()  # 预测正确个数
        prediction = np.append(prediction,pre_test) #保存预测结果---混淆矩阵
        list(map(lambda j: y_fea[j].extend(model.get_fea()[j].cpu().detach().numpy()),range(len(y_fea)))) #保存每一层特征---tsne

    test_accuracy = correct_test / (len(loader_test.dataset)*loader_test.dataset[0].num_nodes)

    print('test accuracy：{}/{}({:.2f}%)'.format(correct_test,len(loader_test.dataset*loader_test.dataset[0].num_nodes),100*test_accuracy))
    print('all train time：{}(s/100epoch)'.format(np.array(train_time).sum()))

    if args.visualization == True:

        visualization_confusion(loader_test=loader_test,prediction=prediction)  #混淆矩阵   实际值 、预测值(预测标签)

        for num,fea in enumerate(y_fea):
            visualization_tsne(loader_test=loader_test,y_feature=np.array(fea),classes=output_dim) #t-SNE可视化  fea --- y_fea

        plt.show()


