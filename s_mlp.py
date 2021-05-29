from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from dataloader import dataLoader
from tools import plot_confusion_matrix


def s_mlp():
    file_path = "dataset/League of Legends.csv"
    X_train, X_test, y_train, y_test = dataLoader(file_path)

    clf = MLPClassifier(activation='relu', solver='adam', alpha=0.0001, hidden_layer_sizes=(214, 214),max_iter=5000)

    clf = clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    print(classification_report(y_test, predict))
    plot_confusion_matrix(confusion_matrix(y_test, predict), classes=range(2), title='confusion matrix')
    print(confusion_matrix(y_test, predict))
    print("acc:", accuracy_score(y_test, predict))


if __name__ == '__main__':
    s_mlp()
