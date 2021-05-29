import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from MyDataSet import MyDataSet
from MLP import MLP
import visdom
from tools import plot_confusion_matrix
import sklearn.metrics as m

batchsz = 32
lr = 1e-3  # 学习率
epoches = 560
torch.manual_seed(1234)
file_path = "dataset/League of Legends.csv"

# 读取数据
train_db = MyDataSet(file_path, mode='train')
val_db = MyDataSet(file_path, mode='val')
test_db = MyDataSet(file_path, mode='test')

train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True)
val_loader = DataLoader(val_db, batch_size=batchsz)
test_loader = DataLoader(test_db, batch_size=batchsz)
viz = visdom.Visdom()


# 计算正确率
def evaluate(model, loader):
    """
    :param model: 网络模型
    :param loader: 数据集
    :return: 正确率
    """
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


# 计算分类的各种指标
def test_evaluate(model, loader):
    """
       :param model: 网络模型
       :param loader: 数据集
       :return: 正确率
    """
    y_true = []
    predict = []
    for x, y in loader:
        with torch.no_grad():
            logits = model(x)
            result = logits.argmax(dim=1)
            for i in y.numpy():
                y_true.append(i)
            for j in result.numpy():
                predict.append(j)

    print(classification_report(y_true, predict))
    plot_confusion_matrix(confusion_matrix(y_true, predict), classes=range(2), title='confusion matrix')
    print("混淆矩阵")
    print(confusion_matrix(y_true, predict))
    print("f1-score:{}.".format(m.f1_score(y_true, predict)))
    return accuracy_score(y_true, predict)


def main():
    model = MLP(16, 2)  # 初始化模型
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 设置Adam优化器
    criterion = nn.CrossEntropyLoss()    # 设置损失函数

    best_epoch, best_acc = 0, 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))   # visdom画loss图
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))  # visdom画val_acc图

    for epoch in range(epoches):
        for step, (x, y) in enumerate(train_loader):
            # x:[b,16] ,y[b]
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        viz.line([loss.item()], [epoch], win='loss', update='append')

        if epoch % 5 == 0:

            val_acc = evaluate(model, val_loader)
            # train_acc = evaluate(model,train_loader)
            print("epoch:[{}/{}]. val_acc:{}.".format(epoch, epoches, val_acc))

            # print("train_acc", train_acc)
            viz.line([val_acc], [epoch], win='val_acc', update='append')
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), 'best.mdl')

    print('best acc:{}. best epoch:{}.'.format(best_acc, best_epoch))
    model.load_state_dict(torch.load('best.mdl'))
    print("loaded from ckpt!")

    test_acc = test_evaluate(model, test_loader)
    print("test_acc:{}".format(test_acc))


if __name__ == '__main__':
    main()
