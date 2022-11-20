##################################################
## A data visualisation script investigating the 
## results of transforming and combining data
##################################################
## Author: Elouan Gros
## Last Updated: Nov 20 2022, 18:38
##################################################

# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 

from pipeline import featurePipeline

# Load the training data
train_data = pd.read_csv("../data/train.csv", index_col=False)
train_data_engineered, _ = featurePipeline(train_data, None, False)

# repartition of tweets sent wrt time of day
f, ax = plt.subplots(figsize=(10, 8))
f.suptitle('Tweets sent wrt time of day', fontsize=16)
ax.ticklabel_format(style="plain")
ax.hist(train_data["TimeOfDay"], bins=24)
ax.set_ylabel('tweets')
ax.set_xlabel('time of day')
plt.savefig("../figs/feature-engineering/TOD_repartition.png")

f, ax = plt.subplots(figsize=(10, 8))
f.suptitle('Retweets wrt time of day', fontsize=16)
ax.ticklabel_format(style="plain")
plt.plot(train_data["TimeOfDay"], train_data['retweets_count'], "r+")
ax.set_ylabel('retweets')
ax.set_xlabel('time of day')
plt.savefig("../figs/feature-engineering/RT_vs_TOD.png")


rt_per_hour = [
    train_data[train_data["TimeOfDay"] == i]["retweets_count"]
    for i in range(24)
]
labels = [
    '{}h'.format(i) for i in range(24)
]
# print(rt_per_hour)
f, ax = plt.subplots(figsize=(10, 8))
f.suptitle('retweets sent wrt time of day', fontsize=16)
ax.ticklabel_format(style="plain")
plt.boxplot(rt_per_hour, vert=False, patch_artist=True, labels=labels)
ax.set_ylabel('retweets')
ax.set_xlabel('time of day')
plt.savefig("../figs/feature-engineering/RT_vs_TOD_mustache.png")

log_rt_per_hour = [
    np.log(train_data[train_data["TimeOfDay"] == i]["retweets_count"] + 1)
    for i in range(24)
]
# print(rt_per_hour)
f, ax = plt.subplots(figsize=(10, 8))
f.suptitle('LOG retweets sent wrt time of day', fontsize=16)
ax.ticklabel_format(style="plain")
plt.boxplot(log_rt_per_hour, vert=False, patch_artist=True, labels=labels)
ax.set_ylabel('retweets')
ax.set_xlabel('time of day')
plt.savefig("../figs/feature-engineering/log_RT_vs_TOD_mustache.png")
