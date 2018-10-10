# !/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd, numpy as np, os, sys, math, time
import matplotlib.pyplot as plt
import random
import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn import preprocessing


"""
models in Propensity Score
"""
class psModels():
    """
    Args:

    """
    def __init__(self):
        self.version = "1.1"

    """
    logistic regression
    Args:
        X: pandas.DataFrame or ndarray
        y: pandas.Series or List or ndarray
    Returns:
        proScore: ndarray
    """
    def logisticReg(self, X, y):
        lr = LogisticRegression()
        lr.fit(X, y)
        proScore = [1 / (1 + _) for _ in np.exp(-np.dot(X, lr.coef_.T)[:, 0] - lr.intercept_[0])]
        return proScore
    """
    k means
    Args:

    Return:

    """
    def clusterByKmeans(self, df):
        df = df.sort_values(by='logP').reset_index(drop=True)
        model = KMeans(n_clusters=3, init="k-means++")
        model.fit(df[['logP']])
        df['cluster'] = model.labels_
        mean0 = np.average(df[df.cluster==0]['logP'])
        mean1 = np.average(df[df.cluster==1]['logP'])
        mean2 = np.average(df[df.cluster==2]['logP'])
        assist = pd.DataFrame([[0,mean0], [1, mean1], [2, mean2]], columns=['code', 'value'])
        assist = assist.sort_values(by='value').reset_index(drop=True)
        left = df[df.cluster==assist.loc[0, 'code']]
        mid = df[df.cluster==assist.loc[1, 'code']]
        right = df[df.cluster==assist.loc[2, 'code']]
        return (left, mid, right, model.cluster_centers_)
    """
    calculate slope
    Args:

    Return:
    """
    def calSlope(self, X, y, slopName):
        lr = LinearRegression()
        lr.fit(X, y)
        return (lr.coef_[0], np.round(lr.coef_[0]/np.var(X['logP'])), lr.score(X, y))

"""
class used for propensity score
propensity score matching, inverse propensity score weighting, covariant adjustments in regression
"""
class PropensityScore():
    """
    Args:
        dims: in genearl, propensity score is used for 2 dimentions
    """
    def __init__(self, dims=2):
        self.dims = dims
        self.psValueName = "psValue"
        # regression model methods
        self.regModels = {
        "logistic regression" : "logisticReg"
        }

    """
    Args:
        dataset数据集合
    """ 
    def generalPSValue(self, dataset, features, label, method='logistic regression'):
        m = psModels()
        if (method=='logistic regression'):
            dataset[self.psValueName] = m.logisticReg(dataset[features], dataset[label])
        else:
            print ("没有相应的方法")
            return None
        return dataset

    """
    propensity score matching
    Args:

    Return:
        Dataset after matching
    """
    def PropensityScoreMatching(self, dataset, label, treatName, thredhold):
        dataset = dataset.sort_values(by=self.psValueName).reset_index(drop=True)
        dataset['index'] = list(dataset.index)
        indexDataset = dataset[['index', label, self.psValueName]]
        # search by forward
        indexDataset['index_a'] = None
        indexDataset['ps_a'] = None
        indexDataset[indexDataset['label'] == treatName]['ps_a'] = indexDataset[self.psValueName]
        indexDataset[indexDataset['label'] == treatName]['index_a'] = indexDataset['index']
        indexDataset['ps_a'] = indexDataset.ps_a.fillna(method='ffill')
        indexDataset['index_a'] = indexDataset.index_a.fillna(method='ffill')
        indexDataset['ps_a'] = indexDataset.ps_a.fillna(3.0)
        # search by backward
        indexDataset['index_b'] = None
        indexDataset['ps_b'] = None
        indexDataset[indexDataset['label'] == treatName]['ps_b'] = indexDataset[self.psValueName]
        indexDataset[indexDataset['label'] == treatName]['index_b'] = indexDataset['index']
        indexDataset['ps_b'] = indexDataset.ps_b.fillna(method='bfill')
        indexDataset['index_b'] = indexDataset.index_b.fillna(method='bfill')
        indexDataset['ps_b'] = indexDataset.ps_b.fillna(3.0)
        indexDataset['distance_a'] = np.abs(indexDataset['ps_a'] - indexDataset[self.psValueName])
        indexDataset['distance_b'] = np.abs(indexDataset['ps_b'] - indexDataset[self.psValueName])
        indexDataset['min_distance'] = np.where(indexDataset['distance_a'] > indexDataset['distance_b'], indexDataset['distance_b'], indexDataset['distance_a'])
        indexDataset['min_index'] = np.where(indexDataset['distance_a'] > indexDataset['distance_b'], indexDataset['index_b'], indexDataset['index_a'])
        indexDataset = indexDataset[indexDataset.min_distance <= thredhold]
        matchedIndexs = list(indexDataset['index'])
        matchedIndexs.extend(list(indexDataset['min_index']))
        return dataset.iloc[matchedIndexs, :]



if __name__ == '__main__':
os.chdir("D:/Report/定价评估")
dataset = pd.read_csv("ori_dataset.csv", index_col=0)
add_dataset = pd.read_csv("add_dataset.csv", index_col=0)
dataset = pd.merge(dataset, add_dataset, on=['skuid', 'sale_ord_dt'], how='inner')
dataset['promotion'] = (dataset['cx_suit_amount'] + dataset['cx_zp_amount'] + dataset['cx_mjmz_amount'] + dataset['cx_dq_amount']) * 1.0 / dataset['red_price'] / dataset['sale_qtty']

"""
plt.scatter(dataset_test[dataset_test.psValue>=0.5]['logP'], dataset_test[dataset_test.psValue>=0.5]['logQ'], color='blue')
plt.scatter(dataset_test[dataset_test.psValue<0.5]['logP'], dataset_test[dataset_test.psValue<0.5]['logQ'], color='red')
"""
# 寻找一个合适的点


returnResult = []
timeStampName = "pics_%s" % int(time.time())
os.mkdir(timeStampName)
skuList = [2047572, 7744807, 1152054, 3048207, 7588711, 3337486, 7588699, 5309662, 7089794, 4950830, 6974110, 4586109, 3355789, 4551403, 7474435, 4689325, 7089824, 5290378, 5792060, 3670111, 7250673, 4605365, 4039471, 2297156, 4094728, 7011704, 5342836, 5876555, 3466756, 7037106, 6840289, 4491723, 1823355, 7531380, 4896842, 5971028, 4126351, 1766544, 4568612, 4819514, 3025902, 5867913, 3994381, 4641723, 2800965, 3603919, 5673852, 3983642, 4475385, 4738245, 5175271, 4334678, 5230976, 4815365, 4181067, 4073004, 5029906, 4363044, 3833413, 5746426, 4491777, 1877093, 4973409, 6558507, 1190232, 4624936, 5967984, 2746186, 4342305, 7182155, 1674862, 1611808, 859781, 4454393, 4431233, 5824693, 2792966, 4420220, 6154947, 5454183, 885412, 903328, 3597418, 3468449, 318470, 2268584, 7127944, 6855535, 3337472, 3167363, 3396391, 2273705, 1865279, 7748935, 5059694, 3625179, 5894552, 8021647, 3690390, 5810057]
for i in skuList:
    try:
        skuId = list(dataset_test.skuid)[0]
        dataset_test = dataset[dataset.skuid == i]
        promothred = np.percentile(dataset_test.promotion, 80)
        dataset_test['flag'] = np.where(dataset_test.promotion >= promothred, 1, 0)
        dataset_test['fixIndex'] = dataset_test.index
        ps = PropensityScore(dims=2)
        ps.generalPSValue(dataset_test,['pv', 'uv'],'flag','logistic regression')
        m = psModels()
        left, mid, right, centers = m.clusterByKmeans(dataset_test)
        mid = mid.sort_values(by='logP').reset_index(drop=True)
        fixIndex = mid.iloc[math.ceil(mid.iloc[:, 0].size/2) - 1, :]['fixIndex']
        # fixIndex定位该点
        fixPs = dataset_test[dataset_test.fixIndex == fixIndex].loc[fixIndex, 'psValue']
        psRange = 0.1
        dataset_test_ps = dataset_test[(dataset_test.psValue >= fixPs - psRange) & (dataset_test.psValue <= fixPs + psRange)]
        sampleRatio = round(dataset_test_ps.iloc[:, 0].size / dataset_test.iloc[:, 0].size, 2)
        slope_ps = m.calSlope(dataset_test_ps[['logP', 'stock', 'trend_pre']], dataset_test_ps['logQ'], 'logP')[0]
        plt.clf()
        plt.title("ps-Scatter:logQ~logP-%s-%s-ratio(%s)" % (skuId, round(slope_ps, 2), sampleRatio))
        plt.scatter(dataset_test_ps['logP'], dataset_test_ps['logQ'])
        plt.savefig("%s/%s-PS(%s)-slope(%s).png" % (timeStampName, skuId, round(fixPs, 2), round(slope_ps, 2)))
        slope_ori = m.calSlope(dataset_test[['logP', 'stock', 'trend_pre']], dataset_test['logQ'], 'logP')[0]
        plt.clf()
        plt.title("ori-Scatter:logQ~logP-%s-%s" % (skuId, round(slope_ps, 2)))
        plt.scatter(dataset_test['logP'], dataset_test['logQ'])
        plt.savefig("%s/%s-ORI(%s)-slope(%s).png" % (timeStampName, skuId, round(fixPs, 2), round(slope_ori, 2)))
        returnResult.append([skuId, fixPs, slope_ps, slope_ori, sampleRatio])
    except:
        pass



