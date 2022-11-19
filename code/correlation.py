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
import numpy as np
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


for name, values in train_data.iteritems():  
    if(name != "text" and name != "urls" and name!= "hashtags"):
        print("Computing {}".format(name))
        #PLot features by label, evolution of correlation
        f, ax = plt.subplots(figsize=(10, 10))
        plt.plot(train_data[str(name)], train_data["retweets_count"], 'bo')
        f.suptitle("Correlation Matrix between {0} and the number of retweets in the orginal Dataset".format(name), fontsize=16)
        plt.xlabel(name)
        print("Correlation Matrix between {0} and the number of retweets in the orginal Dataset".format(name))
        plt.ylabel('Number of retweets')
        print("SAVING FILE...")
        filename  = "../figs/correlation-original/correlation-plotting-{0}.png".format(name)

        plt.savefig(filename, bbox_inches="tight")
        plt.close("all")

for name, values in train_data_engineered.iteritems():  
    if(name != "text"):
        print("Computing {}".format(name))
        #PLot features by label, evolution of correlation 
        f, ax = plt.subplots(figsize=(10, 10))
        plt.plot(train_data_engineered[str(name)], train_data_engineered["retweets_count"], 'bo')
        f.suptitle("Correlation Matrix between {0} and the number of retweets in the orginal Dataset".format(name), fontsize=16)
        plt.xlabel(name)
        plt.ylabel('Number of retweets')
        plt.savefig("../figs/correlation-engineered/correlation-plotting-{0}.png".format(name))