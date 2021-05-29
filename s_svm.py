from sklearn import svm
from dataloader import dataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import sklearn.metrics as m
from tools import plot_confusion_matrix


def s_svm():
    # 获取数据集
    file_name = "dataset/League of Legends.csv"
    X_train, X_test, y_train, y_test = dataLoader(file_name)
    # 初始化支持向量机
    clf = svm.SVC(gamma='scale', C=1.0, kernel='rbf')
    # 训练
    clf.fit(X_train, y_train)
    # 预测
    predict = clf.predict(X_test)
    # 输出评价指标
    print(m.classification_report(y_test, predict))
    plot_confusion_matrix(confusion_matrix(y_test, predict), classes=range(2), title='confusion matrix')
    print("混淆矩阵")
    print(confusion_matrix(y_test, predict))
    print("f1-score:{}.".format(m.f1_score(y_test, predict)))
    print("svm_test_acc:", accuracy_score(y_test, predict))


if __name__ == '__main__':
    s_svm()
