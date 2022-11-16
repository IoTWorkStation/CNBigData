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
import shap
from pickle import load

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = 'Kaiti'
plt.rcParams['axes.unicode_minus'] = False
PIC_PATH = "../models/solve2/"
ti.init(arch=ti.gpu)

# In[2]:


with open('../models/solve1_model.pkl', 'rb') as f:
    model = load(f)
model

# In[2]:


# In[3]:


X = pd.read_csv('../data/solve1_X.csv', index_col=0)
y = pd.read_csv('../data/solve_y.csv', index_col=0)
X

# In[4]:


import pydotplus
from sklearn.tree import export_graphviz

i = 0
for per in model.estimators_:
    dot_data = export_graphviz(per, out_file=None,
                               feature_names=X.columns,
                               class_names=['0', '1'],
                               filled=True, rounded=True,
                               special_characters=True)

    dot_tree_val = dot_data.replace('helvetica', 'MicrosoftYaHei')
    graph = pydotplus.graph_from_dot_data(dot_tree_val)
    i = i + 1
    graph.write_png(PIC_PATH + str(i) + 'tree.png')

# In[5]:


explain = shap.TreeExplainer(model, X)
shap_value = explain.shap_values(X)
shap_value2 = explain(X)

# In[6]:


shap.summary_plot(shap_value, X, show=False)
plt.xlabel('Shap Value')
plt.tight_layout()
plt.savefig(PIC_PATH + 'feature impact.png', dpi=800)

# In[7]:


shap_value2.display_data = X.values
shap.plots.scatter(shap_value2[:, '调和平均温度'][:, 1], color=shap_value2[:, '运行时间'][:, 1], show=False)
plt.tight_layout()
plt.savefig(PIC_PATH + 'temp-time-shap.png', dpi=800)

# In[8]:


shap.dependence_plot("运行时间", shap_value[1], X, show=False)
plt.tight_layout()
plt.savefig(PIC_PATH + 'shap time-npm.png', dpi=800)

# In[9]:


import lime
from lime import lime_tabular

explain_lime = lime_tabular.LimeTabularExplainer(
    training_data=X.to_numpy(),
    feature_names=X.columns,
    class_names=[0, 1],
    mode='classification'
)

# In[10]:


plt.figure(figsize=(16, 9))
exp = explain_lime.explain_instance(data_row=X.iloc[0], predict_fn=model.predict_proba,
                                    num_features=4)
# exp.show_in_notebook()
exp.save_to_file(PIC_PATH + 'choice.png')

# In[11]:


plt.rcParams['font.family'] = ['sans-serif']
plt.figure(figsize=(16, 9))
shap.plots.waterfall(shap_value2[100][:, 0])

# In[13]:


data = pd.concat([X, y], axis=1)
data

# In[15]:


class_group = data.groupby('target')

# In[20]:


description = pd.DataFrame(index=pd.MultiIndex.from_product(
    [data['target'].unique().tolist(),
     ['调和平均温度', '扭矩转速积', '运行时间']]
))


def compute_cov(df):
    return df.std() / df.mean()


count = []
mean = []
median = []
maxin = []
cov = []
maxium = []
minin = []

for classes in data['target'].unique():
    _ = class_group.get_group(classes).drop(['机器质量等级', 'target'], axis=1)
    count.extend(_.count().tolist())
    mean.extend(_.mean().tolist())
    median.extend(_.median().tolist())
    maxin.extend(_.max() - _.min().tolist())
    cov.extend(compute_cov(_).tolist())
    maxium.extend(_.max().tolist())
    minin.extend(_.min().tolist())

description['count'] = count
description['mean'] = mean
description['median'] = median
description['cov'] = cov
description['maxin'] = maxin
description['max'] = maxium
description['min'] = minin

description

# In[21]:


description.to_excel('../data/solve1_description.xlsx')

# In[ ]:
