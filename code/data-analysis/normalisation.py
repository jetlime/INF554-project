##################################################
## A data visualisation script investigating the 
## data distributions of features and the possible
## normalisation techniques 
##################################################
## Author: Paul Houssel
## Last Updated: Nov 19 2022, 21:26
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
train_data = pd.read_csv("../../data/train.csv", index_col=False)
train_data_engineered, _ = featurePipeline(train_data, None, False)
train_data = pd.read_csv("../../data/train.csv", index_col=False)

# INSPECT NORMALISATION - Statuses Count 

f, ax = plt.subplots(figsize=(10, 8))
f.suptitle('Data Distribution (Boxplot) of the  Number of Statuses, including outliers', fontsize=16)
ax.ticklabel_format(style="plain")
sns.boxplot(x=train_data["statuses_count"])
plt.savefig("../../figs/feature-engineering/status-boxplot-outliers.png")


f, ax = plt.subplots(figsize=(10, 8))
ax.ticklabel_format(style="plain")
f.suptitle('Data Distribution (Boxplot) of the  Number of Statuses, not including outliers', fontsize=16)
sns.boxplot(x=train_data["statuses_count"], showfliers=False)
plt.savefig("../../figs/feature-engineering/status-boxplot.png")

f, ax = plt.subplots(figsize=(10, 8))
ax.ticklabel_format(style="plain")
f.suptitle('Density Distribution of the Number of Statuses', fontsize=16)
train_data["statuses_count"].plot.density()
ax.set_xlim(left=-10000)
plt.savefig("../../figs/feature-engineering/status-density-original.png")

# INSPECT NORMALISATION - FOllowers Count

f, ax = plt.subplots(figsize=(10, 8))
f.suptitle('Data Distribution (Boxplot) of the  Number of Followees, including outliers', fontsize=16)
ax.ticklabel_format(style="plain")
sns.boxplot(x=train_data["followers_count"])
plt.savefig("../../figs/feature-engineering/followers_count-boxplot-outliers.png")


f, ax = plt.subplots(figsize=(10, 8))
ax.ticklabel_format(style="plain")
f.suptitle('Data Distribution (Boxplot) of the  Number of Followers, not including outliers', fontsize=16)
sns.boxplot(x=train_data["followers_count"], showfliers=False)
plt.savefig("../../figs/feature-engineering/followers_count-boxplot.png")

f, ax = plt.subplots(figsize=(10, 8))
ax.ticklabel_format(style="plain")
f.suptitle('Density Distribution of the Number of Followers', fontsize=16)
train_data["followers_count"].plot.density()
ax.set_xlim(left=-1000000)
plt.savefig("../../figs/feature-engineering/followers_count-density-original.png")
'''
f, ax = plt.subplots(figsize=(10, 8))
ax.ticklabel_format(style="plain")
f.suptitle('Density Distribution of the text polarity', fontsize=16)
train_data_engineered["text_polarity"].plot.density()
plt.savefig("../../figs/feature-engineering/text_polarity-density-original.png")
''' 
f, ax = plt.subplots(figsize=(10, 8))
ax.ticklabel_format(style="plain")
f.suptitle('Density Distribution of the normalised Number of Followers, using Log Scaling', fontsize=16)
train_data_engineered["followers_count"].plot.density()
ax.set_xlim(left=-1)
plt.savefig("../../figs/feature-engineering/followers_count-density-normalised-log-scaling.png")


# INSPECT NORMALISATION - Favorite Count

f, ax = plt.subplots(figsize=(10, 8))
f.suptitle('Data Distribution (Boxplot) of the  Number of Favorites, including outliers', fontsize=16)
ax.ticklabel_format(style="plain")
sns.boxplot(x=train_data["favorites_count"])
plt.savefig("../../figs/feature-engineering/favorites_count-boxplot-outliers.png")


f, ax = plt.subplots(figsize=(10, 8))
ax.ticklabel_format(style="plain")
f.suptitle('Data Distribution (Boxplot) of the  Number of Favorites, not including outliers', fontsize=16)
sns.boxplot(x=train_data["favorites_count"], showfliers=False)
plt.savefig("../../figs/feature-engineering/favorites_count-boxplot.png")

f, ax = plt.subplots(figsize=(10, 8))
ax.ticklabel_format(style="plain")
f.suptitle('Density Distribution of the Number of Favorites', fontsize=16)
train_data["favorites_count"].plot.density()
ax.set_xlim(left=-10000)
plt.savefig("../../figs/feature-engineering/favorites_count-density-original.png")

f, ax = plt.subplots(figsize=(10, 8))
ax.ticklabel_format(style="plain")
f.suptitle('Density Distribution of the normalised Number of Favorites', fontsize=16)
train_data_engineered["favorites_count"].plot.density()
ax.set_xlim(left=-1)
plt.savefig("../../figs/feature-engineering/favorites_count-density-normalised.png")