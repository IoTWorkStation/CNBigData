import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
import warnings

from skopt import forest_minimize
from skopt.space import space

warnings.filterwarnings('ignore')

data = pd.read_csv('../data/all_data.csv', index_col=0)
dt = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(data.drop(['target', 'class'], axis=1), data['target'],
                                                    test_size=.3, random_state=42)
dt.fit(X_train, y_train)
y = dt.predict(X_test)
y2 = dt.predict(X_train)
score = pd.DataFrame(columns=['score', 'f1_score', 'precision', 'recall'])
scores = [dt.score(X_train, y_train), dt.score(X_test, y_test)]
f1 = [f1_score(y_train, y2), f1_score(y_test, y)]
precision = [precision_score(y_train, y2), precision_score(y_test, y)]
recall = [recall_score(y_train, y2), recall_score(y_test, y)]

score['score'] = scores
score['f1_score'] = f1
score['precision'] = precision
score['recall'] = recall

print(score)


def func_objective(param):
    param = dict(
        zip(
            ['max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes'], param
        )
    )
    dt_opt = DecisionTreeClassifier(**param)
    score = cross_val_score(dt_opt,
                            X=X_train,
                            y=y_train,
                            scoring=make_scorer(f1_score),
                            cv=5)
    return 1 - score.mean()


space_opt = [
    space.Integer(2, 10, name='max_depth'),
    space.Integer(2, 15, name='min_samples_split'),
    space.Integer(2, 15, name='min_samples_leaf'),
    space.Integer(2, 15, name='max_leaf_nodes')
]

res = forest_minimize(func_objective, space_opt)
print(res)

dt_best = DecisionTreeClassifier(max_depth=res.x[0],
                                 min_samples_split=res.x[1],
                                 min_samples_leaf=res.x[2],
                                 max_leaf_nodes=res.x[3]).fit(X_train, y_train)

print(f1_score(y_test, dt_best.predict(X_test)))
print(f1_score(y_train, dt_best.predict(X_train)))
print(confusion_matrix(data['target'], dt.predict(data.drop(['target', 'class'], axis=1))))
