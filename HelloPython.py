from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA ##PCA主成分分析
import jieba
import numpy as np
import pandas as pd
def cont():
    '''
    特征抽取
    :return:
    '''
    # 实例化CountVectorizer
    vector = CountVectorizer()
    # Tf ：term frequency:词的频率  出现的次数
    # idf 逆文档频率 inverse document frequency   log(总文档数量/该词出现的文档的数量)
    # tf * idf
    tfidf = TfidfVectorizer()
    #使用结巴进行分词处理
    data = jieba.cut("你是我的小苹果啊！");
    data3 = jieba.cut("你是我的小苹果啊！");
    data1 = list(data)
    data4 = list(data3)
    data2 = ' '.join(data1)
    data5 = ' '.join(data4)
    print(data2,data5)
    # 调用fit_transform输入并转换数据
    res = vector.fit_transform([data2,data5])
    # 打印结果
    print(res)
    print(vector.get_feature_names())
    print(res.toarray())
    return

def dict():
    '''
    字段数据抽取
    :return:
    '''
    #字典数据抽取  详细用法查看 DictVectorizer API
    from sklearn.feature_extraction import DictVectorizer
    # 实例化字典抽取api
    vector = DictVectorizer(sparse=True)
    # 调用fit_transform
    data = vector.fit_transform([{'city':'北京','code':100},{'city':'上海','code':20},{'city':'广州','code':50},{'city':'深圳','code':66}])
    print(vector.get_feature_names())
    print(data)

def jiebaTest():
    #结巴分词器,使用案例
    import jieba
    data = jieba.cut("你是我的小苹果！")
    data1 = list(data)
    data2 = ' '.join(data1)
    print(data2)

def maxmin():
    '''
    数据特征预处理--------归一化数据处理
                x-min
    公式：X'=  --------      X''=X'*(mx-mi)+mi
                max-min
    注：作用于每一列，max为一列的最大值，min为一列的最小值，那么x''为最终结果，mx,mi分别为指定区间值默认mx为1，mi为0
    :return:
    '''
    maxmin = MinMaxScaler()
    data = maxmin.fit_transform([[90,12,123],[60,123,11],[123,34,55]])
    print(data)
    return

def standar():
    '''
    数据特征预处理--------标准化数据处理
                x - mean
    公式：X' = ----------
                   σ
    注：作用于每一列，mean为平均值，σ为标准差var
                        (x1-mean)^2+(x2-mean)^2+...
    var称为方差，var = ----------------------------- ，σ=√var
                        n(每个特征的样本数)
    其中：方差（考量数据的稳定性）
    :return:
    '''
    standar = StandardScaler();
    data = standar.fit_transform([[90,12,123],[60,123,11],[123,34,55]])
    print(data)
    return

def im():
    '''
    数据特征预处理--------缺失值处理
    :return:
    '''
    im = Imputer(missing_values="NaN",strategy='mean',axis=0)
    data = im.fit_transform([[1,2],[np.nan,3],[7,6]])
    print(data)
    return

def var():
    '''
    数据降维--------特征选择-删除低方差的特征
    :return:
    '''
    var= VarianceThreshold(threshold=0.0)
    data = var.fit_transform([[0,2,3,4],[0,2,3,4],[0,1,1,3]])
    print(data)
    return

def pca():
    '''
    数据降维---------主成分分析进行特征降维
    :return:
    '''
    pca = PCA(n_components=0.93) # n_components 最佳区间在 0.9~0.95之间
    data = pca.fit_transform([[0,2,3,4],[0,2,3,4],[0,1,1,3]])
    print(data)
    return None
"""
函数执行入口
"""
if __name__ == "__main__":
    print("函数执行入口")
    pca()
