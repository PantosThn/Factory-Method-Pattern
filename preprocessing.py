import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


def load_data(data_dir):
    x = pd.read_csv(data_dir + '/x.csv')
    y = pd.read_csv(data_dir + '/y.csv')
    return x, y


def step1(x, y):
    print('running preprocessing step 1')
    return x.values, y.values

def step2(x, y):
    print('running preprocessing step 2')
    return x, y.ravel()


def preprocess(x, y):
    x, y = step1(x, y)
    x, y = step2(x, y)
    return x, y


pipeline_steps = {'normalization': MinMaxScaler,
                  'simple_impute': SimpleImputer,
                  'pca': PCA}


def get_pipeline_steps(steps):
    return [(step, pipeline_steps[step]()) for step in steps]