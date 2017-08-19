import time
import pickle
import math
import numpy as np
import pandas as pd
import gc
import sys
import _thread

import os.path
from sklearn import ensemble
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

FOLDER_NAME = 'data/'
np.random.seed(16)

###### ANIMATED WAIT PRINT###
__stop = False
def __waiting_animation(desc):
    animation = ["[#  ]","[ # ]","[  #]", "[ # ]", "[#  ]"]
    i = 0
    global __stop
    while not __stop:
        time.sleep(0.2)
        sys.stdout.write(desc + "\r" + animation[i % len(animation)])
        sys.stdout.flush()
        i += 1
    sys.stdout.write("\r")
    sys.stdout.flush()
    print("Done!: ",desc)
def __start_animation(msg):
    global __stop
    __stop = False
    _thread.start_new_thread(__waiting_animation, (msg, ))
def __stop_animation():
    global __stop
    __stop = True
    time.sleep(.3)
#############################

class Data:
    def __init__(self, data):
        self.data = data
    def get_data(self):
        return self.data
    def set_targets(self):
        X = self.data.drop(['logerror'], axis=1)
        y = self.data['logerror'].values
        self.X = X
        self.y = y

class DataSet:
    def __init__(self, train:Data, valid:Data, test:Data):
        self.train = train
        self.valid = valid
        self.test = test



def __load_dataset():
    properties = pd.read_csv(FOLDER_NAME+"properties_2016.csv", low_memory=False)
    train = pd.read_csv(FOLDER_NAME+"train_2016_v2.csv")
    return properties, train


##Parsers
def __transactiondate_parser(str_date):
    date_pattern = "%Y-%m-%d"
    date_2016_begin = 1451617200.0
    date_2017_end = 1514689200.0
    epoch = time.mktime(time.strptime(str_date, date_pattern))
    return (epoch - date_2016_begin) / (date_2017_end - date_2016_begin)
def __true_parser(data):
    if data == True:
        return 1
    return 0
def __yes_parser(data):
    if data == "Y":
        return 1
    return 0

##feature importance
def __feature_importance_less(features, labels, min_importance=.2):
    feat_names = features.columns
    model = ensemble.ExtraTreesRegressor(n_estimators=250, max_depth=30, max_features=0.3, n_jobs=-1, random_state=0)
    X = __categorical_parsing_on_column(features).values
    y = labels.values
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    m = np.max(importances)
    for i in range(len(importances)):
        importances[i] = (importances[i] / m)
    output = []
    for i in range(len(indices)):
        if importances[indices[i]] < min_importance:
            output.append(feat_names[indices[i]])

    return output

##One Hot Encoding
def __categorical_parsing(data, cat):
    for i in tqdm(cat):
        if i in data.columns.tolist():
            new_columns = pd.get_dummies(data[i], prefix=i, dummy_na=True)
            data = pd.concat([data.drop(i, axis=1), new_columns], axis=1)
    return data

##Label Encoder
def __categorical_parsing_on_column(merged):
    strcols = merged.select_dtypes([object]).columns.values.tolist()
    for i in tqdm(strcols):
        merged[i] = merged[i].apply(lambda x: str(x))
        encoder = LabelEncoder()
        encoder.fit(merged[i])
        merged[i] = encoder.transform(merged[i])
    return merged


def __columns_droping(data, cols):
    for i in tqdm(cols):
        data = data.drop(i, axis=1)
    return data

##### parsing function
def __parsing_data(data, on_column_encoding, min_importance=-1.0):
    __start_animation("Parsing data")
    data = data.fillna(data.mean())
    data["transactiondate"] = data["transactiondate"].apply(__transactiondate_parser)
    data["hashottuborspa"] = data["hashottuborspa"].apply(__true_parser)
    data["fireplaceflag"] = data["fireplaceflag"].apply(__true_parser)
    data["taxdelinquencyflag"] = data["taxdelinquencyflag"].apply(__yes_parser)
    __stop_animation()

    __start_animation("Trainig feature importance")
    dropping_cols = __feature_importance_less(data.drop(["parcelid", "logerror", "transactiondate"], axis=1),
                                              data["logerror"],
                                              min_importance=-1.0)
    __stop_animation()

    print("Droping columns with low relevance")
    data = __columns_droping(data, cols=dropping_cols)
    if on_column_encoding:
        print("On column labeling")
        data = __categorical_parsing_on_column(data)
    else:
        print("One Hot Encoding with categories")
        data = __categorical_parsing(data, cat=['airconditioningtypeid',
                                              'architecturalstyletypeid',
                                              'buildingclasstypeid',
                                              'decktypeid',
                                              'fips',
                                              'heatingorsystemtypeid',
                                              'pooltypeid10',
                                              'pooltypeid2',
                                              'pooltypeid7',
                                              'propertycountylandusecode',
                                              'propertylandusetypeid',
                                              'regionidcounty',
                                              'regionidcity',
                                              'regionidzip',
                                              'regionidneighborhood',
                                              'storytypeid',
                                              'typeconstructiontypeid',
                                              'propertyzoningdesc'])
    return data



def save_parsed_dataset(dataset_filename, model='OneHotEncoder', min_importance=-1.0):
    
    __start_animation("Loading datasets")
    properties, train = __load_dataset()
    __stop_animation()

    __start_animation("Merge datasets")
    merged = pd.merge(train, properties, on=["parcelid"])
    del properties, train
    __stop_animation()

    collected = gc.collect()
    print ("Garbage collector: collected %d objects." % (collected))
    

    if model == 'OneHotEncoder':
        merged = __parsing_data(merged, on_column_encoding=False, min_importance=min_importance)
    elif model == 'LabelEncoder':
        merged = __parsing_data(merged, on_column_encoding=True, min_importance=min_importance)
    else:
        merged = __parsing_data(merged, on_column_encoding=False, min_importance=min_importance)
    collected = gc.collect()
    print ("Garbage collector: collected %d objects." % (collected))
    
    merged = merged.drop("parcelid", axis=1)

    __start_animation("Dataset scaling")
    scaler = MinMaxScaler()
    avoid_min_max_cols = ["transactiondate", "logerror"]
    scaled = pd.DataFrame(scaler.fit_transform(merged.drop(avoid_min_max_cols,axis=1)), columns=merged.drop(avoid_min_max_cols, axis=1).columns)
    merged = pd.concat([merged[avoid_min_max_cols], scaled], axis=1)
    __stop_animation()

    del scaled, scaler
    collected = gc.collect()
    print ("Garbage collector: collected %d objects." % (collected))

    __start_animation("Permutating index")
    merged = merged.reindex(np.random.permutation(merged.index))
    __stop_animation()

    __start_animation("Saving final parsed dataset into a file")
    data = Data(merged)
    output = open(FOLDER_NAME+dataset_filename, 'wb')
    
    del merged
    collected = gc.collect()
    print ("Garbage collector: collected %d objects." % (collected))
    
    pickle.dump(data, output)
    output.close()
    __stop_animation()



def get_dataset(dataset_filename, model='OneHotEncoder', train=.7, test=.5) -> DataSet:
    if not os.path.isfile(FOLDER_NAME+'/'+dataset_filename):
        save_parsed_dataset(dataset_filename, model=model)
    data = pickle.load(open(FOLDER_NAME+dataset_filename, 'rb'))
    data.set_targets()
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=1.0-train, random_state=22)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=test, random_state=23)
    train = Data(X_train, y_train)
    valid = Data(X_valid, y_valid)
    test = Data(X_test, y_test)
    dataset = DataSet(train, valid, test)
    return dataset



if __name__ == "__main__":
    print ("Label Encoding")
    save_parsed_dataset("LabelEncoder.pkl", model='LabelEncoder')
    print ("Label Encoding Min 20")
    save_parsed_dataset("LabelEncoderMin20.pkl", model='LabelEncoder', min_importance=.2)
    print ("OneHotEncoding")
    save_parsed_dataset("OneHotEncoder.pkl", model='OneHotEncoder')
    print("All datasets generated!")