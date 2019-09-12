from sklearn.feature_extraction import DictVectorizer #分类特征抽取--数据特征提取
from sklearn.feature_extraction.text import CountVectorizer ## 文本特征提取
from sklearn.preprocessing import StandardScaler ### 数据的特征预处理（标准化）
from sklearn.feature_selection import VarianceThreshold ## 数据特征选择
from sklearn.datasets import load_iris  ## 数据集
from sklearn.neighbors import KNeighborsClassifier ### 分类算法----K-近邻
from sklearn.model_selection import train_test_split ###  模型检验-交叉验证    训练集于测试集的分割
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import jieba  ###拆词
import numpy as np

"""
Python学习第一课
"""

def dict():
    """
    分类特征抽取---数据特征抽取
    :return:
    """
    onehot = DictVectorizer() #如果结果不用toarry，请开启sparse=False
    instance = [{'city': '北京','temperature':100},{'city': '上海','temperature':60}, {'city': '深圳','temperature':30}]
    data = onehot.fit_transform(instance).toarray()
    print(data)
    print(onehot.inverse_transform(data))
    return None

def count():
    """
    文本特征抽取
    :return:
    """
    vectorizer = CountVectorizer()
    content = ["life is short,i like python","life is too long,i dislike python"]
    data = vectorizer.fit_transform(content)
    print(vectorizer.get_feature_names())
    print(data.toarray())
    return None

def loadData():
    """
    加载并返回鸢尾属植物数据集
    :return:
    """
    data = load_iris(return_X_y=False)
    print(data.data) ###特征数据数据组
    print(data.target) ## 特征标签数据
    print(data.feature_names) ###特证名
    print(data.target_names) ##标签名
    return None

def k_neighbor():
    """
    分类算法----K-近邻
    :return:
    """
    neigh = KNeighborsClassifier(n_neighbors=3)
    X = np.array([[1,1],[1,1.1],[0,0],[0,0.1]]) ## 训练数据拟合模型
    y = np.array([1,1,0,0])  # 作为X的类别值
    print(neigh.fit(X,y))
    #找到指定点集X的n_neighbors个邻居，return_distance为False的话，不返回距离
    print(neigh.kneighbors(np.array([[1.1,1.1]]),return_distance= False))
    #print(neigh.kneighbors(np.array([[1.1,1.1]]),return_distance= False,an_neighbors=1))
    #预测提供的数据的类标签
    print(neigh.predict(np.array([[0.1,0.1],[1.1,1.1]])))
    #返回测试数据X属于某一类别的概率估计
    print(neigh.predict_proba(np.array([[1.1,1.1]])))
    return None

def knniris():
    """
    K-近邻算法案例分析----------鸢尾花分类
    :return:
    """
    #数据集的获取与分割
    lr = load_iris()
    X_train,x_test,Y_train,y_test=train_test_split(lr.data,lr.target,test_size=0.25)
    #进行标准化处理
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    x_test = std.transform(x_test)
    print("标签：",Y_train)

    #estimator 估计器流程
    knn = KNeighborsClassifier(n_neighbors=3)
    # 得出模型
    knn.fit(X_train,Y_train)

    # 进行预测或者得出精度
    y_predict= knn.predict(x_test)
    print("KNN预测标签类为：",y_predict)
    score = knn.score(x_test,y_test)
    print("测试分数：",score)

    # 通过网格搜索，n_neighbors为参数列表
    param = {"n_neighbors": [3, 5, 7]}
    gs = GridSearchCV(knn,param_grid=param,cv=10)
    # 建立模型
    gs.fit(X_train,Y_train)
    print("从训练集中学习模型参数:",gs)
    #预测数据
    print("预测分数：",gs.score(x_test,y_test))

    # 分离模型的精确度和召回率
    print("每个类别的精确率与召回率：",classification_report(y_test,y_predict,target_names=lr.target_names))

    return None

def test():
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    labels = ['爱情片','爱情片','动作片','动作片']
    knn = KNeighborsClassifier(n_neighbors=3)
    s = knn.fit(group,labels)
    print(s)
    test = [[11,20]]
    data = knn.predict(test)
    print(data)
    data1 = knn.score(test,['爱情片'])
    print(data1)
    return None

if __name__ == "__main__":
    print("程序开始运行！！")
    test()
