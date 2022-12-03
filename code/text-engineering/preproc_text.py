# script to create the bag of words of the hashtag feature to finally encoding with an one-hot encoder
# this script was also used to experiment different dimensionality reduction algorithms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from sklearn.decomposition import PCA


def gen_dataframe(data):
    vectorizer = TfidfVectorizer(max_features=100, stop_words=stopwords.words('french'))
    text_vectorised = vectorizer.fit_transform(data['text'])
    df = pd.DataFrame(data=text_vectorised.todense(), columns=vectorizer.get_feature_names())
    return df

if __name__ == "__main__":
    train_data = pd.read_csv("../../data/train.csv")

    df = gen_dataframe(train_data)
    print(df.shape)
    print(df)
    print(type(df))