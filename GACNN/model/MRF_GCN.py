from torch import nn
import warnings
import torch
from torch_geometric.nn import  ChebConv, BatchNorm
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiChev(torch.nn.Module):   # MRF
    def __init__(self, input_dim):
        super(MultiChev, self).__init__()
        self.scale_1 = ChebConv(input_dim,400,K=1)   # 通道1  K=1，2，3
        self.scale_2 = ChebConv(input_dim,400,K=2)
        self.scale_3 = ChebConv(input_dim,400,K=3)

    def forward(self, x, edge_index,edge_weight ):  # 特征、邻接矩阵、边权重
        scale_1 = self.scale_1(x, edge_index,edge_weight )
        scale_2 = self.scale_2(x, edge_index,edge_weight )
        scale_3 = self.scale_3(x, edge_index,edge_weight )
        return torch.cat([scale_1,scale_2,scale_3],1)   # 聚合连接3个通道  1200

class MultiChev_B(torch.nn.Module):
    def __init__(self, input_dim):
        super(MultiChev_B, self).__init__()
        self.scale_1 = ChebConv(input_dim,100,K=1)
        self.scale_2 = ChebConv(input_dim,100,K=2)
        self.scale_3 = ChebConv(input_dim,100,K=3)
    def forward(self, x, edge_index,edge_weight ):
        scale_1 = self.scale_1(x, edge_index,edge_weight )
        scale_2 = self.scale_2(x, edge_index,edge_weight )
        scale_3 = self.scale_3(x, edge_index,edge_weight )
        return torch.cat([scale_1,scale_2,scale_3],1)  #1200


class MRF_GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MRF_GCN, self).__init__()

        self.conv1 = MultiChev(input_dim).to(device)
        self.bn1 = BatchNorm(1200)
        self.conv2 = MultiChev_B(400 * 3).to(device)  # 400*3通道
        self.bn2 = BatchNorm(300)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Linear(300,  output_dim)

    def forward(self, data):

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        #print(edge_weight)
        self.f1 = x


        x = self.conv1(x, edge_index, edge_weight =  edge_weight)
        x = self.bn1(x)
        x = self.relu(x)
        self.f2 = x

        x = self.conv2(x, edge_index, edge_weight =  edge_weight)
        x = self.bn2(x)

        self.f3 = x

        #x = x.view(x.size(0), -1) # x = x.view(batchsize, -1)   -1指自动分配列数
        x = self.layer1(x)
        self.f4 = x

        out = F.log_softmax(x, dim=1)

        return out

    def get_fea(self):
        return [self.f1,self.f2,self.f3,self.f4]