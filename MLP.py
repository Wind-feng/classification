import torch.nn as nn
import torch.nn.functional as F
import torch


class MLP(nn.Module):
    def __init__(self, input_num, output_num):
        """
        :param input_num: 输出的节点数
        :param output_num: 输出的节点数
        """
        super(MLP, self).__init__()
        hidden1 = 214   # 第二层节点数
        hidden2 = 214   # 第三层节点数
        self.fc1 = nn.Linear(input_num, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_num)
        # 使用dropout防止过拟合
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = torch.relu(self.fc1(x))
        out = self.dropout(out)
        out = torch.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out


