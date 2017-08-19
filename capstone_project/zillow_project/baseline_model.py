import gc
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from zillow_project.data import Data

print('Loading data ...')
######################################################
######################################################
######### ONE HOT ENCODING ###########################
######################################################
######################################################

train = pd.read_csv('data/train_2016_v2.csv')
prop = pd.read_csv('data/properties_2016.csv')
sample = pd.read_csv('data/sample_submission.csv')

print('Binding to float32')

for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.dtype("float32"):
        prop[c] = prop[c].astype(np.dtype("float32"))

print('Creating training set ...')

df_train = train.merge(prop, how='left', on='parcelid')

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()
del train, prop, sample; gc.collect()


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=.3, random_state=22)
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=.5, random_state=23)

print('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
d_test =xgb.DMatrix(x_test, label=y_test)
del x_train, x_valid; gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

y_pred = clf.predict(d_test)

print("BASELINE W/O DATA PREPROCESSING MAE: ", np.abs(y_pred-y_test).mean())

del y_pred, d_test, d_train, d_valid, clf
gc.collect()

######################################################
######################################################
######### ONE HOT ENCODING ###########################
######################################################
######################################################


data = pickle.load(open("data/OneHotEncoder.pkl", 'rb'))
data.set_targets()

x_train, x_test, y_train, y_test = train_test_split(data.X, data.y, test_size=.3, random_state=22)
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=.5, random_state=23)
print('Building DMatrix...')

del data; gc.collect()

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
d_test =xgb.DMatrix(x_test, label=y_test)
del x_train, x_valid; gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

y_pred = clf.predict(d_test)
print("BASELINE ONE HOT ENCODING MAE: ", np.abs(y_pred-y_test).mean())



del y_pred, d_test, d_train, d_valid, clf
gc.collect()

######################################################
######################################################
######### LABEL ENCODING ###########################
######################################################
######################################################


data = pickle.load(open("data/LabelEncoder.pkl", 'rb'))
data.set_targets()

x_train, x_test, y_train, y_test = train_test_split(data.X, data.y, test_size=.3, random_state=22)
x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=.5, random_state=23)
print('Building DMatrix...')
del data; gc.collect()

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
d_test =xgb.DMatrix(x_test, label=y_test)
del x_train, x_valid; gc.collect()

print('Training ...')

params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

y_pred = clf.predict(d_test)
print("BASELINE LABEL ENCODING MAE: ", np.abs(y_pred-y_test).mean())