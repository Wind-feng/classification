import torch
import random, csv
import numpy as np
from torch.utils.data import Dataset
from sklearn import preprocessing


class MyDataSet(Dataset):
    def __init__(self, root, mode):
        super(MyDataSet, self).__init__()

        self.mode = mode
        self.root = root
        label = []
        with open(root, 'r') as f:
            reader = csv.reader(f)
            result = list(reader)
            del result[0]
            random.shuffle(result)
            for i in range(len(result)):
                del result[i][0]
                del result[i][0]
                label.append(int(result[i][0]))
                del result[i][0]
        result = np.array(result, dtype=np.float)
        # result = preprocessing.scale(result).tolist()
        result = preprocessing.StandardScaler().fit_transform(result).tolist()

        # result = preprocessing.MinMaxScaler().fit_transform(result).tolist()
        assert len(result) == len(label)
        self.labels = label
        self.datas = result

        if mode == 'train':  # 60%
            self.datas = self.datas[:int(0.6 * len(self.datas))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val':
            self.datas = self.datas[int(0.6 * len(self.datas)):int(0.8 * len(self.datas))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:
            self.datas = self.datas[int(0.8 * len(self.datas)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        # idx~[0~len(data)]
        data, label = self.datas[idx], self.labels[idx]
        data = torch.tensor(data)
        label = torch.tensor(label)
        return data, label


def main():
    file_path = "dataset/League of Legends.csv"
    db = MyDataSet(file_path, 'train')
    x,y = next(iter(db))
    print(x.shape)
    print(y.shape)


if __name__ == '__main__':
    main()
