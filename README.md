### ml_sort

#### 介绍
本实验使用机器学习算法中的支持向量机、高斯朴素贝叶斯、多项式朴素贝叶斯、多层感知器模型根据League of Legends中的数据
做出比赛结果的预测

#### 目录结构
1. 数据集存放在dataset目录下，名称为League of Legends.csv,该数据集总共48651条数据。
2. result存放的是测试数据集在训练好的模型下的预测结果以及命令行的输出结果。
3. 文件best.mdl为多层感知器保存的模型参数

#### 运行

1. 贝叶斯模型[包括高斯朴素贝叶斯以及多项式朴素贝叶斯]
```
python s_bayes.py
```
2. 支持向量机模型
```
python s_svm.py
```
3.  多层感知机模型
```
python train_mlp.py
```
#### 各个代码功能介绍
1.  dataloader.py 负责从csv中读取数据，并将数据集随机划分。
2.  MyDataSet.py 实现自pytorch的DataSet为MLP模型划分数据
3.  MLP.py 利用pytorch实现的的多层感知器的网络结构
4.  s_bayes.py 贝叶斯算法
5.  s_svm.py 支持向量机算法
6.  tools.py 实现混淆矩阵绘图功能
```
详细的实现过程解释，请参考代码中的注释。
```
项目地址:https://gitee.com/windclub/ml_sort.git





