##################################################
## A data visualisation script investigating the 
## correlation between features for the original 
## dataset and the new dataset 
##################################################
## Author: Paul Houssel
## Last Updated: Nov 19 2022, 21:22
##################################################

# import packages
import pandas as pd
from scipy.interpolate import Rbf
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 

from pipeline import featurePipeline

# Load the training data
train_data = pd.read_csv("../data/train.csv", index_col=False)
train_data_engineered, _ = featurePipeline(train_data, None, False)
train_data = pd.read_csv("../data/train.csv", index_col=False)
# COMPUTE CORRELATION MATRIX 

# run correlation matrix and plot
f, ax = plt.subplots(figsize=(10, 10))
corr = train_data.corr()
sns.heatmap(corr,annot=True, mask=np.zeros_like(corr, dtype=np.bool),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
f.suptitle('Correlation Matrix between features in the orginal Dataset', fontsize=16)
plt.savefig('../figs/correlation-original/correlation-matrix.png')


# run correlation matrix and plot on the engineered features

f, ax = plt.subplots(figsize=(10, 10))
corr = train_data_engineered.corr()
sns.heatmap(corr,annot=True, mask=np.zeros_like(corr, dtype=np.bool),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
f.suptitle('Correlation Matrix between features after Feature Engineering', fontsize=16)
plt.savefig('../figs/correlation-engineered/correlation-matrix.png')

fig = plt.figure(figsize=(15,6))
i = 1
for name, values in train_data.iteritems():  
    if(name != "text" and name != "urls" and name!= "hashtags" and name!="retweets_count" and name!="mentions"):
        print("Computing {}".format(name))
        #PLot features by label, evolution of correlation
        ax = fig.add_subplot(2,4,i)
        # cubic smooth line
        ax.plot(train_data[str(name)], train_data["retweets_count"], 'bo', markersize=1.5, label='_nolegend_')
        if(i==1):
            train_data.groupby(str(name))["retweets_count"].mean().ewm(com=2).mean().plot(color='red', ax=ax, label="Mean Exponentially weighted Smoothing on the mass center", lw=1)
        else:
            train_data.groupby(str(name))["retweets_count"].mean().ewm(com=2).mean().plot(color='red', ax=ax, lw=1, label='_nolegend_')
        if(i==1):
            plt.ylabel('Number of retweets')
        plt.xlabel(name)    
        i += 1

fig.legend(shadow=True, loc="lower right")
print("SAVING FILE...")
filename  = "../figs/correlation-original/correlation-plotting.png"
plt.tight_layout()
plt.savefig(filename)
plt.close("all")


fig = plt.figure(figsize=(15,6))
i = 1
for name, values in train_data_engineered.iteritems():  
    if(name != "text" and name!="retweets_count"):
        print("Computing {}".format(name))
        #PLot features by label, evolution of correlation
        ax = fig.add_subplot(2,4,i)
        # cubic smooth line
        ax.plot(train_data_engineered[str(name)], train_data_engineered["retweets_count"], 'bo', markersize=1.5, label='_nolegend_')
        if(i==1):
            train_data_engineered.groupby(str(name))["retweets_count"].mean().ewm(com=1).mean().plot(color='red', ax=ax, label="Mean Exponentially weighted Smoothing on the mass center", lw=1)
        else:
            train_data_engineered.groupby(str(name))["retweets_count"].mean().ewm(com=1).mean().plot(color='red', ax=ax, lw=1, label='_nolegend_')
        if(i==1):
            plt.ylabel('Number of retweets')
        plt.xlabel(name)    
        i += 1

fig.legend(shadow=True, loc="lower right")
print("SAVING FILE...")
filename  = "../figs/correlation-engineered/correlation-engineered.png"
plt.tight_layout()
plt.savefig(filename)
plt.close("all")



f, ax = plt.subplots(figsize=(10, 10))
sns.lmplot(x='followers_count', y='retweets_count', data=train_data_engineered)
f.suptitle('Correlation Matrix between features in the orginal Dataset', fontsize=16)
plt.savefig('../figs/correlation-original/correlation-matrix.png')
