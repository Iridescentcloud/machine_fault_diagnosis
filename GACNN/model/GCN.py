import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self,input_dim,output_dim):
        super(GCN, self).__init__()

        self.GConv1 = GCNConv(input_dim,512)
        self.bn1 = BatchNorm(512)

        self.GConv2 = GCNConv(512,300)
        self.bn2 = BatchNorm(300)

        self.linear = nn.Sequential(
            nn.Linear(300, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr  # 节点特征值 、 边连接 、 边权重

        self.f1 = x  # 原始特征(时域输入时为时域特征，频域输入时为频域特征)

        x = self.GConv1(x, edge_index,edge_weight)
        x = self.bn1(x)
        x = F.relu(x)

        self.f2 = x  # 第一层特征

        x = self.GConv2(x, edge_index,edge_weight)
        x = self.bn2(x)
        x = F.relu(x)

        self.f3 = x  # 第二层特征

        x = self.linear(x)

        self.f4 = x  # 全连接层特征

        out = F.log_softmax(x,dim=1)

        return out

    def get_fea(self):
        return [self.f1,self.f2,self.f3,self.f4]