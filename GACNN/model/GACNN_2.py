import torch
from torch import nn
import warnings
from torch_geometric.nn import  ChebConv, BatchNorm
import torch.nn.functional as F
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ChebyNet(torch.nn.Module):
    def __init__(self,input_dim,output_dim):
        super(ChebyNet, self).__init__()

        self.ChebyConv1 = ChebConv(input_dim,512,K=2)
        self.bn1 = BatchNorm(512)

        self.ChebyConv2 = ChebConv(512, 300, K=2)
        self.bn2 = BatchNorm(300)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.ChebyConv1(x, edge_index,edge_weight)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.ChebyConv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class CAE(nn.Module):
    def __init__(self,input_channel,gap_size,stride=1):
        super(CAE, self).__init__()
        b = 1
        gama = 2
        kernel_size = int(abs((math.log(input_channel, 2) + b) / gama))
        if kernel_size % 2:
            kernel_size = kernel_size    # 如果卷积核大小是偶数，就使用它
        else:
            kernel_size = kernel_size + 1   # 如果卷积核大小是奇数就变成偶数

        self.GAP = nn.AdaptiveAvgPool1d(gap_size)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        b, c, _ = x.size()  # (批次数，通道，大小)
        y = self.GAP(x)   # 压缩
        y = y.transpose(-1, -2)
        y = self.conv (y)
        y = y.transpose(-1,-2)
        y = self.sigmoid(y)
        y = x * y.expand_as(x)
        return y

class CNN_1D(nn.Module):
    def __init__(self,in_channel):
        super(CNN_1D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 8, kernel_size=3,stride=1, padding=1),# 16, 26 ,26  卷积大小为kernel_size*in_channels； padding=1输入的每一条边补充0的层数
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3,stride=1),  # 32, 24, 24
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv1d(16,32, kernel_size=3,stride=1),  # 32, 24, 24
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(255, 75),  #    32 x 90 = 2880
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True))
    def forward(self,data):
        x = data.x.unsqueeze(dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)   #320,32,126
        #x = self.fc(x)
        return x


class GACNN_2(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(GACNN_2, self).__init__()

        self.GCN = ChebyNet(input_dim, output_dim)   # model = ChebyNet(input_dim, output_dim).to(device)
        self.CNN = CNN_1D(in_channel=1).to(device)
        self.CAE = CAE(input_channel=32, gap_size=1, stride=1).to(device)
        self.drop1 = nn.Dropout(p=0.2)
        self.relu = nn.ReLU(inplace=True)
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(4332, 1200),    # CWRU:4332
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Linear(1200, 512),
        )

        self.fc2 = nn.Linear(512, output_dim)

    def forward(self,data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr   # 1024
        self.f1 = x

        # GCN
        x = self.GCN(data)

        # CNN
        y = self.CNN(data)
        #y = self.drop1(y)

        # 拉平
        y = y.view(y.size(0), -1)

        # 聚合
        out = torch.cat([x, y], 1)
        out = out.unsqueeze(dim=1)

        out = self.drop1(out)
        out = self.CAE(out)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        # 分类
        x = self.fc(out)
        x = self.fc2(x)

        self.f2 = x
        out = F.log_softmax(x, dim=1)

        return out

    def get_fea(self):
        return [self.f1, self.f2]


