import pandas as pd
import matplotlib.pyplot as plt
from XGBOOST import *
import os
import numpy as np
import re


splits = {'train': 'simplified/train-00000-of-00001.parquet',
          'validation': 'simplified/validation-00000-of-00001.parquet',
          'test': 'simplified/test-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/google-research-datasets/go_emotions/" + splits["train"])
df_val = pd.read_parquet("hf://datasets/google-research-datasets/go_emotions/" + splits["validation"])
df_test = pd.read_parquet("hf://datasets/google-research-datasets/go_emotions/" + splits["test"])

# Drop data points with more than one label
df_train_simplified = df_train[df_train["labels"].apply(lambda x: len(x) == 1)]
df_val_simplified = df_val[df_val["labels"].apply(lambda x: len(x) == 1)]
df_test_simplified = df_test[df_test["labels"].apply(lambda x: len(x) == 1)]


def downsample(data):
    majority = data[data['labels'] == 27]
    minority = data[data['labels'] != 27]

    # Downsample majority class to match minority class size
    majority_downsampled = majority.sample(n=int(len(majority)/6), random_state=42)

    # Combine downsampled majority class with minority class
    balanced_data = pd.concat([majority_downsampled, minority])

    return balanced_data


'''
df_test_simplified = downsample(df_test_simplified)
df_train_simplified = downsample(df_train_simplified)
df_val_simplified = downsample(df_val_simplified)
'''


run_xgboost_pipeline(df_train_simplified,df_val_simplified,df_test_simplified)

