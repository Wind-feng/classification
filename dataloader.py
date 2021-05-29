import csv
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split

file_path = "dataset/League of Legends.csv"


def dataLoader(filename):
    """
    :param filename:
    :param filepath:
    :return: len(result) = 48652
             len(result[0]) = 19
    """
    label = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
        del result[0]
        # print(result[1][0])
        for i in range(len(result)):
            del result[i][0]
            del result[i][0]
            label.append(int(result[i][0]))
            del result[i][0]

    result = np.array(result, dtype=np.float)
    result = preprocessing.StandardScaler().fit_transform(result)
    label = np.array(label, dtype=np.int)
    X_train, X_test, y_train, y_test = train_test_split(result, label, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    dataLoader(file_path)
