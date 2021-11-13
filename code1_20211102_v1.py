# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: cook
#     language: python
#     name: cook
# ---

# +
import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# -

# # Description
# ## 1. accuracy
#     - v1 : 순정
#         DecisionTree : 0.69
#         RandomForest : 0.71
#         GradientBoost : 0.69
#     - v2 : 변수선택 및 타입조절
#         DecisionTree : 0.69
#         RandomForest : 0.71
#         GradientBoost : 0.69

# ## data load and wrangling

train = pd.read_csv('./input/235713_신용카드 사용자 연체 예측 AI 경진대회_data/open/train.csv')
test = pd.read_csv('./input/235713_신용카드 사용자 연체 예측 AI 경진대회_data/open/test.csv')
sub = pd.read_csv('./input/235713_신용카드 사용자 연체 예측 AI 경진대회_data/open/sample_submission.csv')

train.shape
# train.isnull().sum()

# +
train.nunique()
train.dtypes

# train_x.isnull().sum()
# -

# ### 1. EDA

nm_y = ['credit']
nm_num = ['child_num', 'income_total', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL', 
          'work_phone', 'phone', 'email', 'family_size', 'begin_month']
nm_str = train.columns.drop(nm_num)
nm_str

# +
train[nm_str].head(5)
train[nm_str].nunique()

train['car'].value_counts()
# train['income_type'].value_counts()
# train['edu_type'].value_counts()
# train['family_type'].value_counts()
# train['house_type'].value_counts()

test['car'].value_counts()
# test['income_type'].value_counts()
# test['edu_type'].value_counts()
# test['family_type'].value_counts()
# test['house_type'].value_counts()
# -

train[nm_str].nunique()
# test[nm_str].nunique()

# #### 1-1. numeric

train[nm_num].head()

train[nm_num].describe()

# 1. 'work_phone', 'phone', 'email'은 최소가 0, 최대가 1인 값이다. -> categorical로 처리.
#
# 2. 'FLAG_MOBIL'는 모든 값이 1이다. -> Feature로서 의미가 없으므로 제외
#
# 3. 'child_num'은 Q3가 1인데, max는 19이다. 19가 이상치가 아닌지 의심해볼 필요가 있다.
#
# 4. 'family_size'도 마찬가지이다.
#
# 5. 'DAYS_EMPLOYED' 변수는 양수가 없을텐데 최대값이 양수다.

#확인
train['email'].value_counts()
train['FLAG_MOBIL'].value_counts()

train['child_num'].value_counts(), test['child_num'].value_counts()
# 'child_num'이 14인 3명과, 19인 1명은 제외한다.

train['family_size'].value_counts(), test['family_size'].value_counts() 
# 'family_size'가 9인 사람과 2인 사람을 보고, 제거한다.

# +
train.sort_values(by = ['DAYS_EMPLOYED'], axis = 0, ascending = False).head(15)
# 'DAYS_EMPLOYED'가 365243인 사람이 많은듯

tmp = train.loc[train['DAYS_EMPLOYED'] != 365243] # 4438명
tmp['DAYS_EMPLOYED'].describe()
# 찾아보니, 양수는 무직을 의미한다.

# 무직/
# 직장 만5년차 미만/
# 직장 만5년이상 만10년 미만/
# 직장 만10년차이상 
# 으로 4 그룹으로 먼저 나누어보기로 함.


# +
# train
cond = [
    (train['DAYS_EMPLOYED'] >= 0),
    (train['DAYS_EMPLOYED'] <= 0) & (train['DAYS_EMPLOYED'] > -5*365),
    (train['DAYS_EMPLOYED'] <= -5*365) & (train['DAYS_EMPLOYED'] > -10*365),
    (train['DAYS_EMPLOYED'] <= -10*365)
]

vals = ['None', 'Emp_under5', 'Emp_5to10', 'Emp_over10']
train['employed_grp'] = np.select(cond, vals)

# test
cond = [
    (test['DAYS_EMPLOYED'] >= 0),
    (test['DAYS_EMPLOYED'] <= 0) & (test['DAYS_EMPLOYED'] > -5*365),
    (test['DAYS_EMPLOYED'] <= -5*365) & (test['DAYS_EMPLOYED'] > -10*365),
    (test['DAYS_EMPLOYED'] <= -10*365)
]

vals = ['None', 'Emp_under5', 'Emp_5to10', 'Emp_over10']
test['employed_grp'] = np.select(cond, vals)
# -

train[['DAYS_EMPLOYED', 'employed_grp']].head(40)
train['employed_grp'].value_counts().reset_index()

pltdf = train['employed_grp'].value_counts().reset_index()
plt.bar(pltdf['index'], pltdf['employed_grp'])

train[(train['child_num'] >= 14) | (train['family_size'] >= 9)]

#1 outliers 제거
# 'child_num'이 14이상인 사람은 test set에 없다. 따라서 outlier 취급한다.
# 'family_size'가 9 이상인 사람은 test set에 없다. 따라서 outlier 취급한다.
train = train[(train['child_num'] < 14) & (train['family_size'] < 9)]
test = test[(test['child_num'] < 14) & (test['family_size'] < 9)]
train.shape # why 6개 안지워짐..?

# +
#2 index
train_id = train['index']
test_id = test['index']

#3 target variable
train_y = train['credit']

#4 features
train_x = train.drop(['index', 'credit', 'occyp_type', 'DAYS_EMPLOYED', 'FLAG_MOBIL'], axis = 1)
test_x = test.drop(['index', 'occyp_type', 'DAYS_EMPLOYED', 'FLAG_MOBIL'], axis = 1)

# +
#5 변수타입 정리
nm_num = ['child_num', 'income_total', 'DAYS_BIRTH', 
          'family_size', 'begin_month']
nm_str = train_x.columns.drop(nm_num)

print('categorical : ', nm_str, 
      '\n',
      'numeric :', nm_num)
# -

train_x.shape, test_x.shape

# ### 2. scaler

train_x_cl = train_x.copy()
test_x_cl = test_x.copy()

test_x[nm_num].columns

ss = StandardScaler()
train_x_cl[nm_num] = pd.DataFrame(ss.fit_transform(train_x[nm_num]), columns = train_x[nm_num].columns)
test_x_cl[nm_num] = pd.DataFrame(ss.transform(test_x[nm_num]), columns = test_x[nm_num].columns)

le = LabelEncoder()
for i in nm_str:
    le.fit(train_x[i])
    train_x_cl[i] = le.transform(train_x[i])
    test_x_cl[i] = le.transform(test_x[i])

train_x_cl.head()
type(train_x_cl)

train_x_cl.shape

# +
train_x_cl.isnull().sum()

train_x_cl[train_x_cl['child_num'].isnull() == True]
# -

# validation set
train_x_cl_t, train_x_cl_v, train_y_t, train_y_v = train_test_split(train_x_cl, train_y,
                 random_state = 21, test_size = 0.2)

# ## modeling
# ### 1. DecisionTree

md_d1 = DecisionTreeClassifier(max_depth=2, 
                               random_state=2021)
md_d1.fit(train_x_cl_t, train_y_t)

pred_d1 = md_d1.predict(train_x_cl_v)

accuracy_score(train_y_v, pred_d1)

confusion_matrix(train_y_v, pred_d1)

plot_confusion_matrix(md_d1, train_x_cl_t, train_y_t)

plot_confusion_matrix(md_d1, train_x_cl_v, train_y_v)

# +
# from sklearn.tree import export_graphviz

# export_graphviz(md_d1, out_file = 'tree.dot')
# -

pred_d1

# ### 2. RandomForest

md_r1 = RandomForestClassifier(n_estimators=300, 
                               random_state=2021)
md_r1.fit(train_x_cl_t, train_y_t)

pred_r1 = md_r1.predict(train_x_cl_v)

accuracy_score(pred_r1, train_y_v)

# ### 3. Gardient Boosting

md_g1 = GradientBoostingClassifier(random_state=2021)
md_g1.fit(train_x_cl_t, train_y_t)

pred_g1 = md_g1.predict(train_x_cl_v)
accuracy_score(pred_g1, train_y_v)



# ## predict test set

md_d2 = DecisionTreeClassifier(max_depth=2)
md_d2.fit(train_x_cl, train_y)

pred_d2 = md_d2.predict(test_x_cl)

pred_d2




