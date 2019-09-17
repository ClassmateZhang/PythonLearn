from sklearn.datasets import load_iris,fetch_20newsgroups,load_boston  #获取数据集
from sklearn.model_selection import train_test_split  ## 训练集与测试集的获取
li = load_iris()
# print("获取数据特征值")
# print(li.data)
# print("获取目标值")
# print(li.target)
# print(li.DESCR)

#注意返回值，训练集 train   x_train,y_train     测试集 test x_test, y_test
# x_train,x_test,y_train,y_test = train_test_split(li.data,li.target,test_size=0.25)
# print("训练集特征值和目标值：",x_train,y_train)
# print("测试集特征值和目标值：",x_test,y_test)

##获取新闻数据
# news = fetch_20newsgroups(data_home='D:/download/',subset='all')
# print(news.data)
# print(news.target)

