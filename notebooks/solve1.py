#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
PIC_PATH = "../models/solve1/"
ti.init(arch=ti.gpu)

# In[3]:


data = pd.read_excel('../data/train data.xlsx')
columns = data.columns.tolist()
columns[4] = '机械工作温度'
data.columns = columns
data

# In[4]:


data.iloc[:, 1].nunique()

# In[5]:


sns.countplot(data.iloc[:, -2])
plt.savefig(PIC_PATH + '是否发生故障.png', dpi=800)

# In[6]:


sns.countplot(data.iloc[:, -1][data.iloc[:, -1] != 'Normal'])
plt.savefig(PIC_PATH + '具体故障类别.png', dpi=800)

# In[7]:


columns = data.columns

data.iloc[:, 2: -2].hist(figsize=(16, 8))

# In[8]:


kl.missingval_plot(data)

# In[9]:


for column in columns[3: -2]:
    kl.dist_plot(data[column])
    plt.tight_layout()
    plt.savefig(PIC_PATH + column + ' density.png', dpi=800)

# In[10]:


kl.corr_plot(data[columns[3: -2]])
plt.tight_layout()
plt.savefig(PIC_PATH + 'pearsonr.png', dpi=800)


# In[11]:


def cov(df: pd.DataFrame):
    return df.std() / df.mean()


# In[12]:


cov(data)

# In[13]:


class_group = data.groupby('是否发生故障')

# In[14]:


data[columns[-2:]].value_counts()

# In[15]:


outlier = class_group.get_group(1)
normal = class_group.get_group(0)

# In[16]:


ques = outlier[outlier['具体故障类别'] == 'Normal']
ques

# In[17]:


sns.countplot(data.iloc[:, 2])
plt.tight_layout()
plt.savefig(PIC_PATH + '机器质量等级', dpi=800)

# In[18]:


sns.countplot(outlier.iloc[:, 2])
plt.tight_layout()
plt.savefig(PIC_PATH + '故障机器质量等级', dpi=800)

# In[19]:


sns.countplot(normal.iloc[:, 2])
plt.tight_layout()
plt.savefig(PIC_PATH + '正常机器质量等级', dpi=800)

# In[20]:


normal.describe()

# In[21]:


normal.skew(), normal.kurtosis()

# In[22]:


outlier.describe()

# In[23]:


outlier.skew(), outlier.kurtosis()

# In[24]:


sns.distplot(normal['转速（rpm）'], label='1')
sns.distplot(outlier['转速（rpm）'], label='0')
plt.legend()
plt.tight_layout()
plt.savefig(PIC_PATH + 'rpm对比.png', dpi=800)

# In[25]:


sns.distplot(normal['扭矩（Nm）'], label='1')
sns.distplot(outlier['扭矩（Nm）'], label='0')
plt.legend()
plt.tight_layout()
plt.savefig(PIC_PATH + 'Nm对比.png', dpi=800)

# In[26]:


sns.distplot(normal['室温（K）'], label='1')
sns.distplot(outlier['室温（K）'], label='0')
plt.legend()
plt.tight_layout()
plt.savefig(PIC_PATH + 'K对比.png', dpi=800)

# In[27]:


sns.distplot(normal['机械工作温度'], label='1')
sns.distplot(outlier['机械工作温度'], label='0')
plt.legend()
plt.tight_layout()
plt.savefig(PIC_PATH + 'K.1对比.png', dpi=800)

# In[28]:


sns.distplot(normal['使用时长（min）'], label='1')
sns.distplot(outlier['使用时长（min）'], label='0')
plt.legend()
plt.tight_layout()
plt.savefig(PIC_PATH + 'time对比.png', dpi=800)

# In[29]:


ques

# In[30]:


sns.pairplot(data[columns[3: -2]])
plt.tight_layout()
plt.savefig(PIC_PATH + '联合分布.png', dpi=800)


# In[31]:


def compute(df):
    record = pd.DataFrame(columns=columns[3:-2])
    record.iloc[:, 0] = (ques['室温（K）'] - df['室温（K）'].mean()) / df['室温（K）'].std()
    # record.iloc[:, 1] = (ques['室温（K）'] - outlier['室温（K）'].mean()) / outlier['室温（K）'].std()
    record.iloc[:, 1] = (ques['机械工作温度'] - df['机械工作温度'].mean()) / df['机械工作温度'].std()
    record.iloc[:, 2] = (ques['转速（rpm）'] - df['转速（rpm）'].mean()) / df['转速（rpm）'].std()
    record.iloc[:, 3] = (ques['扭矩（Nm）'] - df['扭矩（Nm）'].mean()) / df['扭矩（Nm）'].std()
    record.iloc[:, 2] = (ques['使用时长（min）'] - df['使用时长（min）'].mean()) / df['使用时长（min）'].std()
    return record


# In[32]:


normal_record = compute(normal)
outlier_record = compute(outlier)

# In[33]:


normal_record

# In[34]:


outlier_record

# In[35]:


data.drop(ques.index, inplace=True)
data

# In[36]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
target = data.iloc[:, -2]
train = data.iloc[:, 2: -2]
train

# In[37]:


train.iloc[:, 1:] = ss.fit_transform(train.iloc[:, 1:])
train

# In[38]:


level_map = {
    'L': 0,
    'M': 1,
    'H': 2
}

train['机器质量等级'] = train.iloc[:, 0].map(level_map)
train

# In[39]:


from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

model_list = [
    DBSCAN(), IsolationForest(), LocalOutlierFactor()
]
score_list = []

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=.3, random_state=42)

for model in model_list:
    y_hat = model.fit_predict(train, target)
    score_list.append(f1_score(target, y_hat, average='micro'))
score_list

# In[40]:


test = pd.concat([train, target], axis=1)
test

# In[41]:


kl.dist_plot(np.log(1 + data['转速（rpm）']))

# In[42]:


np.corrcoef(data['扭矩（Nm）'], 1 / data['转速（rpm）']), np.corrcoef(data['室温（K）'], data['机械工作温度'])


# In[43]:


def harmonic_mean(data1):
    lst = []

    for i in data1.values:
        total = 0
        for j in i:
            total += 1 / j
        lst.append(len(i) / total)

    return lst


# In[44]:


new_data = pd.DataFrame()

new_data.index = data['机器编号']
new_data['temp'] = harmonic_mean(data[['室温（K）', '机械工作温度']])
new_data['npm'] = (data['扭矩（Nm）'] * data['转速（rpm）']).tolist()
new_data['level'] = (data['机器质量等级'].map(level_map)).tolist()
new_data['time'] = (data['使用时长（min）']).tolist()
new_data['target'] = (data['是否发生故障']).tolist()
new_data

# In[45]:


kl.corr_plot(new_data)

# In[46]:


kl.dist_plot(new_data['npm'])

# In[47]:


sns.kdeplot(data=new_data, x='temp', hue='target')

# In[48]:


from statsmodels.graphics.mosaicplot import mosaic

mosaic(new_data, ['level', 'target'])

# In[49]:


sns.displot(new_data, x='time', hue='target')

# In[50]:


cluster = new_data.copy()
cluster[['temp', 'npm', 'time']] = ss.fit_transform(cluster[['temp', 'npm', 'time']])

# In[51]:


from sklearn.cluster import KMeans

model_list = [
    KMeans(n_clusters=2), DBSCAN(1), IsolationForest(), LocalOutlierFactor()
]
score_list = []

for model in model_list:
    y_hat = model.fit_predict(cluster.drop('target', axis=1))
    np.where(y_hat <= 0, 0, 1)
    score_list.append(f1_score(new_data['target'], y_hat, average='micro'))
score_list

# In[52]:


import scipy.stats as stats

stats.pointbiserialr(cluster['target'], new_data['npm'])

# In[53]:


stats.pointbiserialr(new_data['target'], new_data['temp'])

# In[54]:


stats.mannwhitneyu(new_data[new_data['target'] == 0],
                   new_data[new_data['target'] == 1],
                   alternative='two-sided')
# cohen's D


# In[55]:


new_group = new_data.groupby('target')
bad = new_group.get_group(1)
good = new_group.get_group(0)

# In[56]:


bad

# In[57]:


label = {
    'temp': '温度调和平均值',
    'npm': '转速扭矩积',
    'time': '使用时间'
}

for col in new_data.columns[:-1]:
    if col != 'level':
        kl.dist_plot(new_data[col])
        plt.savefig(PIC_PATH + col + ' all transform.png', dpi=800)

# In[58]:


for col in ['temp', 'npm', 'time']:
    plt.figure()
    sns.distplot(bad[col], label='1')
    sns.distplot(good[col], label='0')
    plt.xlabel(label[col])
    plt.legend()
    plt.tight_layout()
    plt.savefig(PIC_PATH + col + ' transform.png', dpi=800)

# In[59]:


bad['level'].value_counts()

# In[60]:


bad['level'].value_counts() / new_data['level'].value_counts()


# In[61]:


def ana_level(df1, df2, name):
    sns.distplot(df1[name], label='0')
    sns.distplot(df2[name], label='1')
    plt.legend()
    plt.savefig(PIC_PATH + name + ' level dietubed.png', dpi=800)


# In[62]:


for col in ['temp', 'npm', 'time']:
    plt.figure()
    ana_level(good, bad, col)

# In[63]:


from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

gauss = BayesianGaussianMixture(n_components=2, tol=10)
y_hat = gauss.fit_predict(new_data.drop('target', axis=1))
f1_score(new_data['target'], y_hat)

# In[64]:


plt.figure(figsize=(16, 10))
sns.scatterplot(data=new_data, x='temp', y='npm', size='level', hue='target')
plt.xlabel('调和平均温度')
plt.ylabel('扭矩转速积')
plt.tight_layout()
plt.savefig(PIC_PATH + '温度-扭转速度积分布.png', dpi=800)

# In[65]:


plt.figure(figsize=(16, 10))
sns.scatterplot(data=new_data, x='temp', y='time', size='level', hue='target')
plt.xlabel('调和平均温度')
plt.ylabel('工作时间')
plt.tight_layout()
plt.savefig(PIC_PATH + '时间-扭转速度积分布.png', dpi=800)

# In[66]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 10))
ax = fig.gca(projection='3d')

ax.scatter(good['temp'], good['npm'], good['time'], c='b', marker='o', label='正常')
ax.scatter(bad['temp'], bad['npm'], bad['time'], c='r', marker='x', label='故障')
ax.view_init(elev=45, azim=-45)

ax.set_xlabel('调和平均温度')
ax.set_ylabel('扭矩转速积')
ax.set_zlabel('运转时间')
plt.legend()
plt.tight_layout()
plt.savefig(PIC_PATH + '3D Density.png', dpi=1200)

# In[67]:


from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

GOOD_LENGTH = int(len(good) * .8)
BAD_LENGTH = int(len(bad) * .8)
train = pd.concat([good.iloc[:GOOD_LENGTH], bad.iloc[:BAD_LENGTH]], axis=0)
plt.tight_layout()
test = pd.concat([good.iloc[GOOD_LENGTH:], bad.iloc[BAD_LENGTH:]], axis=0)

# In[68]:


from sklearn.metrics import confusion_matrix


# In[68]:


# In[69]:


def gaussian_kernel(x1, x2, sigma):
    return np.exp(-np.power(x1 - x2, 2) / (2 * sigma ** 2))


# In[70]:


npm = ss.fit_transform(new_data['npm'].to_numpy().reshape(-1, 1))
temp = ss.fit_transform(new_data['temp'].to_numpy().reshape(-1, 1))
time = ss.fit_transform(np.sqrt(2 * np.log1p(new_data['time'].to_numpy().reshape(-1, 1))) *
                        np.cos(2 * np.pi * new_data['time'].to_numpy().reshape(-1, 1)))
gauss_feature = gaussian_kernel(npm, temp, 3)
gauss_feature

# In[71]:


sns.distplot(time)

# In[71]:


# In[72]:


fig = plt.figure(figsize=(16, 10))
ax = fig.gca(projection='3d')
boolean = new_data['target'] == 0

ax.scatter(temp[boolean], npm[boolean], gauss_feature[boolean], c='b', marker='o', label='正常')
ax.scatter(temp[~boolean], npm[~boolean], gauss_feature[~boolean], c='r', marker='x', label='故障')
ax.view_init(elev=45, azim=-45)

ax.set_xlabel('调和平均温度')
ax.set_ylabel('扭矩转速积')
ax.set_zlabel('高斯融合')
plt.legend()
ax.view_init(elev=45, azim=-45)
plt.tight_layout()
plt.savefig(PIC_PATH + '3D Gaussian.png', dpi=1200)

# In[73]:


transform = new_data.copy()
transform['temp'] = temp
transform['npm'] = npm
transform['time'] = time
transform['gaussian'] = gauss_feature

plt.figure(figsize=(16, 9))
sns.scatterplot(data=transform, x='time', y='gaussian', size='level', hue='target')

# In[74]:


from sklearn.mixture import BayesianGaussianMixture

numberic = ['temp', 'npm', 'gaussian']

bayes = BayesianGaussianMixture(tol=10, n_components=2, reg_covar=1e-1)
bayes.fit(transform.drop('target', axis=1))
y_hat = bayes.predict(transform.drop('target', axis=1))

# In[75]:


f1_score(y_true=transform['target'], y_pred=y_hat, average='macro')

# In[76]:


from sklearn.metrics import roc_auc_score, recall_score, precision_score

roc_auc_score(y_true=transform['target'],
              y_score=bayes.predict_proba(transform.drop('target', axis=1))[:, 1])

# In[77]:


recall_score(y_true=transform['target'], y_pred=y_hat)

# In[78]:


confusion_matrix(transform['target'], y_hat)

# In[79]:


from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from skopt import forest_minimize, space


def func_objective(param):
    param = dict(
        zip(['covariance_type', 'tol', 'reg_covar', 'max_iter'], param)
    )
    bayes_opt = BayesianGaussianMixture(n_components=2, **param)
    score = cross_val_score(bayes_opt,
                            X=transform.drop('target', axis=1),
                            y=transform['target'],
                            scoring=make_scorer(roc_auc_score),
                            cv=5)
    return 1 - score.mean()


# In[80]:


space_opt = [
    space.Categorical(categories=['full', 'tied', 'diag', 'spherical'], name='covariance_type'),
    space.Real(1e-3, 15, name='tol'),
    space.Real(1e-3, 15, name='reg_covar'),
    space.Integer(50, 200, name='max_iter')
]

# In[81]:


res = forest_minimize(func_objective, space_opt)
res.x

# In[82]:


bayes_best = BayesianGaussianMixture(n_components=2,
                                     covariance_type='tied',
                                     tol=res.x[1],
                                     reg_covar=res.x[2],
                                     max_iter=res.x[3])
bayes_best.fit(transform.drop('target', axis=1))
y_hat = bayes_best.predict(transform.drop('target', axis=1))

# In[83]:


f1_score(y_true=transform['target'], y_pred=y_hat, average='macro')

# In[84]:


roc_auc_score(y_true=transform['target'],
              y_score=bayes_best.predict_proba(transform.drop('target', axis=1))[:, 1])

# In[85]:


confusion_matrix(transform['target'], y_hat)

# In[86]:


new_data['gauss'] = gauss_feature
new_data.index = range(len(new_data))
# new_data.drop('机器编号', axis=1, inplace=True)
new_data

# In[87]:


X = new_data.drop('target', axis=1)
y = new_data['target']
X

# In[88]:


boolean = boolean.to_numpy()
true = X[boolean]
false = X[~boolean]

# In[89]:


result_model = pd.DataFrame(index=['precision', 'recall', 'accuracy', 'F1_score'])

# In[90]:


from sklearn.tree import DecisionTreeClassifier

out_data = false.sample(280, replace=False)
test_data = false.drop(out_data.index, axis=0)

score_list = []
model_list = []
for i in range(25):
    train_false = out_data.sample(250, replace=False)
    test_false = false.drop(train_false.index)
    sample = true.sample(2 * len(train_false), replace=False)
    X_train = pd.concat([train_false, sample], axis=0)
    y_train = y.iloc[X_train.index]
    dt = DecisionTreeClassifier(
        class_weight={0: 1, 1: 2},
        random_state=42
    )
    dt.fit(X_train, y_train)
    score_list.append(dt.score(X_train, y_train))
    model_list.append(dt)

model_list

# In[91]:


from sklearn.base import ClassifierMixin
from sklearn.tree import BaseDecisionTree
from sklearn.metrics import accuracy_score


class Voting(ClassifierMixin, BaseDecisionTree):

    def __init__(self, ensemble: list):
        self.ensemble = ensemble

    def predict(self, x):
        result_df = pd.DataFrame()
        for c in range(len(self.ensemble)):
            m = self.ensemble[c]
            result_df[c] = m.predict(x)
        return result_df.mode(axis=1)

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X))

    def precision(self, x_test, y_test):
        return precision_score(y_test, self.predict(x_test))

    def recall(self, x_test, y_test):
        return recall_score(y_test, self.predict(x_test))

    def f1(self, x_test, y_test, method='macro'):
        return f1_score(y_test, self.predict(x_test), average=method)


# In[92]:


voting = Voting(model_list)
voting.predict(test_data)

# In[93]:


voting_score = []
voting_score.append(voting.precision(X, y))
voting_score.append(voting.recall(X, y))
voting_score.append(voting.score(X, y))
voting_score.append(voting.f1(X, y))
result_model['Voting_Pro'] = voting_score
voting_score

# In[94]:


confusion_matrix(y, voting.predict(X))

# In[95]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=25)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.3)

rf.fit(X_train, y_train)
predict = rf.predict(X)
pre_test = rf.predict(X_test)
rf.score(X_test, y_test)


# In[96]:


def create_model(model_name, df, target):
    X_train, X_test, y_train, y_test = train_test_split(df, target, random_state=42, test_size=.3)
    model_name.fit(X_train, y_train)
    y_hat = model_name.predict(X)
    print(f1_score(y_test, model_name.predict(X_test)))
    return [
        precision_score(target, y_hat),
        recall_score(target, y_hat),
        model_name.score(X, target),
        f1_score(target, y_hat)
    ]


# In[97]:


result_model

# In[98]:


result_model['RF_pro'] = create_model(rf, X, y)
result_model['DT_pro'] = create_model(DecisionTreeClassifier(), X, y)

# In[99]:


confusion_matrix(y, predict)

# In[99]:


# In[99]:


# In[100]:


X = X.drop('gauss', axis=1)

true = X[boolean]
false = X[~boolean]

train_false = false.sample(280, replace=False)
test_false = false.drop(train_false.index)

score_list2 = []
model_list2 = []
for i in range(15):
    sample = true.sample(3 * len(train_false), replace=False)
    X_train = pd.concat([train_false, sample], axis=0)
    y_train = y.iloc[X_train.index]
    dt = DecisionTreeClassifier(class_weight={0: 1, 1: 5})
    dt.fit(X_train, y_train)
    score_list2.append(dt.score(X_train, y_train))
    model_list2.append(dt)

# In[101]:


voting2 = Voting(model_list2)

# In[102]:


result_model['Voting'] = [
    voting2.precision(X, y),
    voting2.recall(X, y),
    voting2.score(X, y),
    voting2.f1(X, y)
]

# In[129]:


result_model['RF'] = create_model(RandomForestClassifier(n_estimators=25), X, y)
result_model['DT'] = create_model(DecisionTreeClassifier(), X, y)

# In[104]:


boolean2 = voting.predict(new_data.drop('target', axis=1))
boolean2 = (boolean2 == 1).to_numpy()
boolean2

# In[105]:


second = new_data[boolean2]
second

# In[106]:


second['target'].value_counts()

# In[107]:


plt.figure(figsize=(16, 9))
sns.scatterplot(x='temp', y='npm', data=second, size='level', hue='target')

# In[108]:


plt.plot(second['gauss'], 'o')

# In[109]:


forecast = pd.read_excel('../data/forecast.xlsx', index_col=0)
forecast

# In[110]:


fore = pd.DataFrame()

fore['temp'] = harmonic_mean(forecast[['室温（K）', '室温（K）.1']])
fore['npm'] = (forecast['转速（rpm）'] * forecast['扭矩（Nm）']).tolist()
fore['level'] = forecast['机器质量等级'].map(level_map).tolist()
fore['time'] = forecast.iloc[:, -2].tolist()

fore

# In[111]:


# result_model.drop('RF_pro_s', axis=1, inplace=True)
result_model

# In[112]:


# result = voting.predict(fore)
# result


# In[113]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
best_model = RandomForestClassifier(n_estimators=25).fit(X_train, y_train)
best_model.score(X_test, y_test)

# In[114]:


best_model.score(X, y)

# In[115]:


result = best_model.predict(fore)
forecast['是否发生故障'] = result
forecast.to_excel('../data/forecast.xlsx')

# In[130]:


forecast['是否发生故障'].value_counts()

# In[116]:


new_data['class'] = data['具体故障类别'].tolist()
to_csv = new_data[new_data['class'] != 'Normal']
to_csv

# In[117]:


to_csv.to_csv('../data/bad_data.csv')
data

# In[118]:


csv_data = pd.DataFrame()
csv_data['level'] = data['机器质量等级'].map(level_map)
csv_data['temp_1'] = data['室温（K）'].tolist()
csv_data['temp_2'] = data['机械工作温度'].tolist()
csv_data['efficient'] = list(data['转速（rpm）'] * data['扭矩（Nm）'])
csv_data['time'] = data['使用时长（min）'].tolist()
csv_data['target'] = data['是否发生故障'].tolist()
csv_data['class'] = data['具体故障类别'].tolist()

csv_data.to_csv('../data/all_data.csv')

# In[119]:


csv_data

# In[120]:


X.columns = ['调和平均温度', '扭矩转速积', '机器质量等级', '运行时间']
X

# In[121]:


import eli5

eli5.show_weights(best_model, feature_names=X.columns.tolist())


# In[122]:


def func_objective(param):
    param = dict(
        zip(
            ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes'], param
        )
    )
    dt_opt = RandomForestClassifier(**param)
    score = cross_val_score(dt_opt,
                            X=X_train,
                            y=y_train,
                            scoring=make_scorer(f1_score),
                            cv=5)
    return 1 - score.mean()


space_opt = [
    space.Integer(15, 200, name='n_estimators'),
    space.Integer(2, 10, name='max_depth'),
    space.Integer(2, 15, name='min_samples_split'),
    space.Integer(2, 15, name='min_samples_leaf'),
    space.Integer(2, 15, name='max_leaf_nodes')
]

res = forest_minimize(func_objective, space_opt, verbose=True)

# In[123]:


res

# In[124]:


param = dict(
    zip(
        ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes'], res.x
    )
)

create_model(
    RandomForestClassifier(**param),
    X,
    y
)

# In[125]:


res.x

# In[125]:


# In[126]:


RES = pd.DataFrame(res.x_iters)
RES.columns = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes']
RES.to_excel('../data/iter.xlsx')
RES

# In[126]:


# In[127]:


from pickle import dump

with open('../models/solve1_model.pkl', 'wb') as f:
    dump(best_model, f)

X.to_csv('../data/solve1_X.csv')
y.to_csv('../data/solve_y.csv')

# In[128]:


from sklearn.linear_model import LogisticRegression

create_model(LogisticRegression(), X, y)

# In[128]:


# In[128]:


# In[128]:
