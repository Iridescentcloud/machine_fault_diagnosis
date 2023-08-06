import torch.nn as nn
import torch.nn.functional as F

# nn.Module
class CNN_1D(nn.Module):  #类
    def __init__(self, in_channel, out_channel):
        super(CNN_1D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 8, kernel_size=3,padding=1),  # 16, 26 ,26  卷积大小为kernel_size*in_channels； padding=1输入的每一条边补充0的层数
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2,stride=2))   #池化

        self.layer2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3),  # 32, 24, 24
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))  # 32, 12,12     (24-2) /2 +1

        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3),  # 32, 24, 24
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        # 全连接层
        self.layer4 = nn.Sequential(
            nn.Linear(32*126, 1024),   # 64通道  * 30  1920
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True))

        self.fc = nn.Linear(512, out_channel)



    def forward(self, data):
        x = data.x.unsqueeze(dim=1)  # shape: （样本数，1，样本长度）

        self.f1 = x.reshape(x.shape[0],x.shape[1] * x.shape[2])  # 原始特征(时域输入时为时域特征，频域输入时为频域特征) --- 将通道维度展平
       # (640,1024)
        x = self.layer1(x)

        #self.f2 = x.reshape(x.shape[0], x.shape[1] * x.shape[2]) #第一层卷积特征
        self.f2 = x.view(x.size(0), -1)

        x = self.layer2(x)

        #self.f3 = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  # 第二层卷积特征
        self.f3 = x.view(x.size(0), -1)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)

        x = self.layer4(x)
        x = self.fc(x)

        #self.f7 = x
        self.f7 = x.reshape(x.shape[0], x.shape[1])  # 全连接层特征

        out = F.log_softmax(x,dim=1)

        return out
        #return x

    def get_fea(self):
        return [self.f1,self.f2,self.f3]


