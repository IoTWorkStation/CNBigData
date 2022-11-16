#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.base
import taichi as ti
import klib as kl
import warnings
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = 'Kaiti'
plt.rcParams['axes.unicode_minus'] = False
PIC_PATH = "../models/solve3/"
DATA_PATH = "../data/"
ti.init(arch=ti.gpu)

# In[2]:


data = pd.read_csv(DATA_PATH + 'bad_data.csv', index_col=0)
data = data.drop('gauss', axis=1)
data

# In[3]:


data_a = pd.read_csv(DATA_PATH + 'all_data.csv', index_col=0)
new_data = data_a[data_a['target'] == 1]
new_data

# In[4]:


sns.countplot(data['class'])
plt.savefig(PIC_PATH + 'class density.png', dpi=800)

# In[5]:


data.drop('target', axis=1, inplace=True)
class_group = data.groupby('class')

# In[6]:


feature_map = {
    'temp': '调和平均温度',
    'npm': '扭矩转速积',
    'time': '运行时间'
}

for col in ['temp', 'npm', 'time']:
    plt.figure(figsize=(8, 5))
    for classes in data['class'].unique():
        _ = class_group.get_group(classes)
        sns.distplot(_[col], label=classes)
    plt.xlabel(feature_map.get(col))
    plt.legend()
    plt.savefig(PIC_PATH + col + 'feature density.png', dpi=800)

# In[7]:


class_all = data['class'].unique()
class_all

# In[8]:


for classes in class_all:
    print(classes)
    print(class_group.get_group(classes).describe())

# In[9]:


new_data['temp_diff'] = new_data['temp_2'] - new_data['temp_1']
kl.corr_plot(new_data.drop('target', axis=1))

# In[10]:


new_data

# In[11]:


plt.figure(figsize=(8, 5))
new_group = new_data.groupby('class')
for classes in data['class'].unique():
    _ = new_group.get_group(classes)
    sns.distplot(_['temp_diff'], label=classes)
    plt.legend()
plt.savefig(PIC_PATH + 'temp_diff.png', dpi=800)

# In[12]:


data['温差'] = new_data['temp_diff'].tolist()

data


# In[13]:


def compute_cov(df):
    return df.std() / df.mean()


# In[14]:


description = pd.DataFrame(index=pd.MultiIndex.from_product(
    [data['class'].unique().tolist(),
     ['调和平均温度', '扭矩转速积', '运行时间', '温差']]
))

count = []
mean = []
median = []
maxin = []
cov = []

for classes in data['class'].unique():
    _ = class_group.get_group(classes).drop(['class', 'level'], axis=1)
    count.extend(_.count().tolist())
    mean.extend(_.mean().tolist())
    median.extend(_.median().tolist())
    maxin.extend(_.max() - _.min().tolist())
    cov.extend(compute_cov(_).tolist())

# In[15]:


description['count'] = count
description['mean'] = mean
description['median'] = median
description['maxin'] = maxin
description['cov'] = cov

description

# In[16]:


description.to_csv(DATA_PATH + 'solve3_description.csv')

# In[17]:


from autoviz.AutoViz_Class import AutoViz_Class

temp = data.copy()
temp.columns = ['调和平均温度', '扭矩转速积', '机器质量等级', '运行时间', '故障类别', '温差']
plt.rcParams['font.sans-serif'] = ['Kaiti']
temp['机器质量等级'] = temp['机器质量等级'].astype('category')
temp['运行时间'] = temp['运行时间'].astype('float32')
av = AutoViz_Class()
dft = av.AutoViz(
    filename=None,
    depVar='故障类别',
    dfte=temp,
    chart_format='png',
    save_plot_dir=PIC_PATH
)

# In[18]:


X = temp.drop('故障类别', axis=1)
y = temp['故障类别']

# In[19]:


y

# In[20]:


y_ = y.copy()
# y_ = np.where(y_=='RNF', y_, 'other')
# y_


# In[21]:


des = pd.concat([X, pd.DataFrame(y_, columns=['故障类别'], index=y.index)], axis=1)
des

# In[22]:


columns = des.columns

plt.figure(figsize=(24, 12))
plt.subplot(321)
sns.scatterplot(data=des, x=columns[0], y=columns[1], hue=columns[-1], size=columns[2])
plt.legend(loc='upper left')
plt.subplot(322)
sns.scatterplot(data=des, x=columns[0], y=columns[-2], hue=columns[-1], size=columns[2])
plt.legend(loc='upper left')
plt.subplot(323)
sns.scatterplot(data=des, x=columns[1], y=columns[-2], hue=columns[-1], size=columns[2])
plt.legend(loc='upper left')
plt.subplot(324)
sns.scatterplot(data=des, x=columns[0], y=columns[3], hue=columns[-1], size=columns[2])
plt.legend(loc='upper left')
plt.subplot(325)
sns.scatterplot(data=des, x=columns[1], y=columns[3], hue=columns[-1], size=columns[2])
plt.legend(loc='upper left')
plt.subplot(326)
sns.scatterplot(data=des, x=columns[-2], y=columns[3], hue=columns[-1], size=columns[2])
plt.legend(loc='upper left')
plt.savefig(PIC_PATH + 'other density.png', dpi=1000)

# In[23]:


from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import BayesianGaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM
from lazypredict.Supervised import LazyClassifier

# In[24]:


x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
clf = LazyClassifier(predictions=True)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
models

# In[25]:


y.to_numpy()

# In[26]:


svm = OneClassSVM(tol=8).fit_predict(X)
pd.DataFrame(svm).value_counts()

# In[27]:


_ = np.where(y == 'RNF', 4, y)
_ = pd.DataFrame(index=y.index, data=_)
_[_ == 4].dropna()

# In[28]:


target = pd.DataFrame(index=y.index, data=svm)
target[_ == 4].dropna()

# In[29]:


X

# In[30]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def return_score(model, get_x=X, get_y=y):
    predicts = cross_val_predict(model, get_x, get_y, cv=5)

    _scores = [
        precision_score(y, predicts, average='weighted'),
        recall_score(y, predicts, average='weighted'),
        accuracy_score(y, predicts),
        f1_score(y, predicts, average='weighted')
    ]
    return _scores


# In[31]:


from sklearn.neighbors import KNeighborsClassifier

scores = pd.DataFrame()
X_ = X.drop('机器质量等级', axis=1)
scores['RF'] = return_score(RandomForestClassifier(n_estimators=25), X_, y)
scores['DT'] = return_score(DecisionTreeClassifier(), X_, y)
scores['svm'] = return_score(SVC(), StandardScaler().fit_transform(X_), y)
scores['knn'] = return_score(KNeighborsClassifier(), StandardScaler().fit_transform(X_), y)

scores

# In[32]:


temp[['故障类别', '机器质量等级']].value_counts()

# In[33]:


level0 = temp[temp['机器质量等级'] == 0]
level1 = temp[temp['机器质量等级'] == 1]
level2 = temp[temp['机器质量等级'] == 2]

classes = temp['故障类别'].unique()

percent = pd.DataFrame()


def return_length(class_num, level_num):
    gp = class_group.get_group(class_num)
    return len(gp[gp['level'] == level_num])


for i in classes:
    count = len(class_group.get_group(i))
    percent[i] = [
        return_length(i, 0) / count,
        return_length(i, 1) / count,
        return_length(i, 2) / count
    ]

percent

# In[34]:


X_

# In[35]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

stacking = StackingClassifier(
    estimators=[('rf', RandomForestClassifier(n_estimators=25))],
    final_estimator=LogisticRegression()
)

# In[36]:


from imblearn.over_sampling import ADASYN

boolean1 = y == 'RNF'
boolean2 = y == 'TWF'
sampling = pd.concat([y[boolean1], y[boolean2]], axis=0)
sampling_x = pd.concat([X[boolean1], X[boolean2]], axis=0)
ana = ADASYN()
RNF, RNF_y = ana.fit_resample(sampling_x, sampling)
RNF_y

# In[37]:


RNF.fillna(0, inplace=True)
RNF

# In[38]:


X_new = pd.concat([X, RNF], axis=0)
y_new = pd.concat([y, RNF_y], axis=0)
X_new

# In[39]:


cli = pd.concat([X_new, y_new], axis=1)
cli

# In[40]:


x_train, x_test, y_train, y_test = train_test_split(X_new, y_new, test_size=.3, random_state=42)
model = RandomForestClassifier()
model.fit(x_train, y_train)
model.score(x_test, y_test)

# In[41]:


f1_score(y_test, model.predict(x_test), average='weighted')

# In[42]:


from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix, confusion_matrix

multilabel_confusion_matrix(y_test, model.predict(x_test))

# In[43]:


confusion_matrix(y_test, model.predict(x_test))

# In[44]:


plot_confusion_matrix(
    model,
    x_test,
    y_test
)
plt.tight_layout()
plt.savefig(PIC_PATH + 'test-confusion-matrix.png', dpi=800)

# In[45]:


plot_confusion_matrix(
    model,
    X,
    y
)
plt.tight_layout()
plt.savefig(PIC_PATH + 'data-confusion-matrix.png', dpi=800)

# In[46]:


confusion_matrix(
    y,
    model.predict(X)
)

# In[47]:


res = model.predict(X)

display(
    model.score(X, y),
    precision_score(y, res, average='weighted'),
    recall_score(y, res, average='weighted'),
    f1_score(y, res, average='weighted')
)

# In[48]:


from joblib import dump

dump(model, DATA_PATH + 'solve3.model')

# In[49]:


X


# In[50]:


def harmonic_mean(data1):
    lst = []

    for i in data1.values:
        total = 0
        for j in i:
            total += 1 / j
        lst.append(len(i) / total)

    return lst


level_map = {
    'L': 0,
    'M': 1,
    'H': 2
}


def process(df: pd.DataFrame) -> pd.DataFrame:
    _result = pd.DataFrame()
    _result['调和平均温度'] = harmonic_mean(
        df[['室温（K）', '室温（K）.1']]
    )
    _result['扭矩转速积'] = list(df['转速（rpm）'] * df['扭矩（Nm）'])
    _result['机器质量等级'] = df['机器质量等级'].map(level_map).tolist()
    _result['运行时间'] = list(df['使用时长（min）'].astype('float32'))
    _result['温差'] = list(df['室温（K）.1'] - df['室温（K）'])
    return _result


# In[59]:


forecast = pd.read_excel(DATA_PATH + 'forecast.xlsx')
forecast

# In[60]:


train = forecast[forecast['是否发生故障'] == 1]
train

# In[61]:


pro = process(train)
pro

# In[62]:


target = model.predict(pro)
target

# In[63]:


train

# In[64]:


forecast['故障类别'].loc[train.index]

# In[65]:


forecast['故障类别'] = np.nan
forecast['故障类别'].loc[train.index] = target
forecast

# In[66]:


forecast['故障类别'].fillna('Normal', inplace=True)
forecast.to_excel(DATA_PATH + 'forecast.xlsx', index=False)

# In[ ]:


X.to_csv(DATA_PATH + 'X.csv')
y.to_csv(DATA_PATH + 'y.csv')
