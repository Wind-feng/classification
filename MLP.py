import torch.nn as nn
import torch.nn.functional as F
import torch


class MLP(nn.Module):
    def __init__(self, input_num, output_num):
        super(MLP, self).__init__()
        hidden1 = 214
        hidden2 = 214
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


