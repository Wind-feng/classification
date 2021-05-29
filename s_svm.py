from sklearn import svm
from dataloader import dataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import sklearn.metrics as m
from tools import plot_confusion_matrix


def s_svm():
    file_name = "dataset/League of Legends.csv"
    X_train, X_test, y_train, y_test = dataLoader(file_name)
    clf = svm.SVC(gamma='scale', C=1.0, kernel='rbf')
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)

    print(m.classification_report(y_test, predict))
    plot_confusion_matrix(confusion_matrix(y_test, predict), classes=range(2), title='confusion matrix')
    print("混淆矩阵")
    print(confusion_matrix(y_test, predict))
    print("f1-score:{}.".format(m.f1_score(y_test, predict)))
    print("svm_test_acc:", accuracy_score(y_test, predict))


if __name__ == '__main__':
    s_svm()
