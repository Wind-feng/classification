from sklearn.naive_bayes import GaussianNB, MultinomialNB
from dataloader import dataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tools import plot_confusion_matrix
import sklearn.metrics as m

target_names = ['class 0', 'class 1']


def s_GaussianNB():
    gnb = GaussianNB()
    file_name = "dataset/League of Legends.csv"
    X_train, X_test, y_train, y_test = dataLoader(file_name)
    gnb = gnb.fit(X_train, y_train)
    predict = gnb.predict(X_test)

    print(classification_report(y_test, predict))
    plot_confusion_matrix(confusion_matrix(y_test, predict), classes=range(2), title='confusion matrix')
    print("混淆矩阵")
    print(confusion_matrix(y_test, predict))
    print("f1-score:{}.".format(m.f1_score(y_test, predict)))
    print("GaussianNB_test_acc:", accuracy_score(y_test, predict))


def s_MultinomialNB():
    gnb = MultinomialNB()
    file_name = "dataset/League of Legends.csv"
    X_train, X_test, y_train, y_test = dataLoader(file_name)
    gnb = gnb.fit(X_train, y_train)
    predict = gnb.predict(X_test)

    print(classification_report(y_test, predict))
    plot_confusion_matrix(confusion_matrix(y_test, predict), classes=range(2), title='confusion matrix')
    print("混淆矩阵")
    print(confusion_matrix(y_test, predict))
    print("f1-score:{}.".format(m.f1_score(y_test, predict)))
    print("MultinomialNB_test_acc:", accuracy_score(y_test, predict))


if __name__ == '__main__':
    s_GaussianNB()
    s_MultinomialNB()
