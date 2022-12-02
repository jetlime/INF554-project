import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt 



def gen_BOW(data):
    """generate bag of words"""
    hashtags = [h.strip("[]").split(", ") for h in data['hashtags']]
    BOW = {}
    BOWs = []
    for l in hashtags:
        tmp = {}
        for h in l:
            if len(h) == 0:
                continue
            
            if h in BOW:
                BOW[h] += 1
            else:
                BOW[h] = 1

            if h in tmp:
                tmp[h] += 1
            else:
                tmp[h] = 1

        BOWs.append(tmp)

    shortBOW = dict(sorted(BOW.items(), key=lambda item: item[1])[-100:])
    proc_BOWs = [
        dict(filter(lambda i: i[0] in shortBOW, B.items())) for B in BOWs
        # dict([(a,b/shortBOW[a]) for a,b in B.items() if a in shortBOW]) for B in BOWs
    ]
        
    return shortBOW, proc_BOWs

def gen_BOW2(data, tail=None, descriminate=None):
    """returns the list of words in the bag of words"""
    hashtags = [h.strip("[]").split(", ") for h in data['hashtags']]
    BOW = {}
    for l in hashtags:
        for h in l:
            if len(h) == 0:
                continue
            
            if h in BOW:
                BOW[h] += 1
            else:
                BOW[h] = 1

    if tail is not None:
        return list(map(lambda x: x[0], list(sorted(BOW.items(), key=lambda item: item[1]))[-tail:]))
    if descriminate is not None:
        return list(map(lambda x: x[0], list(sorted(filter(descriminate, BOW.items()), key=lambda item: item[1]))))
    return list(map(lambda x: x[0], list(sorted(BOW.items(), key=lambda item: item[1]))))

def sentence_BOW(data, BOW):
    hashtags = h.strip("[]").split(", ")
    return dict(filter(lambda i: i[0] in shortBOW, B.items()))

def sentence_BOW2(data, BOW):
    """returns bit encoding of hashtag presence"""
    return np.array([1 if i in data else 0 for i in BOW])

def batch_sentence_BOW2(data, BOW):
    ret = np.zeros((data.shape[0], len(BOW)))
    for i in range(data.shape[0]):
        for j in data[i].strip("[]").split(", "):
            if j in BOW:
                ret[i, BOW.index(j)] = 1

    return ret

def gen_BOW_data(data):
    BOW = gen_BOW2(data, None)
    X = [sentence_BOW2(i.strip('[]').split(", "), BOW) for i in data['hashtags']]
    Y = np.floor(np.log(data["retweets_count"]+1))
    return X, Y

def gen_BOW_dataframe(data, tail=None):
    # Only keep the hashtags appearing more than 5 times
    BOW = gen_BOW2(data, tail, descriminate=lambda x:x[1]>5)
    print("BOW is done")
    print(len(BOW))
    X = batch_sentence_BOW2(data["hashtags"], BOW)
    #X = [sentence_BOW2(i.strip('[]').split(", "), BOW) for i in data['hashtags']]
    df = pd.DataFrame(X, columns=BOW)
    df["retweets_count"] = data["retweets_count"]
    return df

def gen_pca(df):
    pca = PCA(n_components=5)
    X = df.drop(['retweets_count'], axis=1)
    pca_features = pca.fit_transform(X,df["retweets_count"])
    return pca_features, pca

def gen_lda(df):
    clf = LDA(n_components=5)
    X = df.drop(['retweets_count'], axis=1)
    df_new = clf.fit_transform(X, df["retweets_count"])
    return df_new, clf

def gen_nmf(df):
    clf = NMF(n_components=5, verbose=2)
    X = df.drop(['retweets_count'], axis=1)
    df_new = clf.fit_transform(X)
    return df_new, clf


if __name__ == "__main__":
    train_data = pd.read_csv("../data/train.csv")
    # tmp = gen_BOW_data(train_data)
    # pprint(tmp[0][-10:])
    # pprint(tmp[1][-10:])
    # pprint(np.max(tmp[1]))
    df = gen_BOW_dataframe(train_data, None)
    print(df.shape)
    df.to_csv("../data/BOW-full.csv")
    #df = pd.read_csv('../data/BOW.csv')

    # NMF
    #df = pd.read_csv('../data/BOW.csv')
    df_new,pca = gen_pca(df)


    plt.bar(
    range(1,len(pca.explained_variance_ratio_)+1),
    pca.explained_variance_ratio_
    )
    

    plt.xlabel('pca Feature')
    plt.ylabel('Explained variance')
    plt.title('Feature Explained Variance')
    plt.savefig('../figs/hashtags/pca.png')
    with open('../figs/hashtags/pca-feature.npy', 'wb') as f:
        np.save(f, pca)
    '''
    # PCA
    df = pd.read_csv('../data/BOW.csv')
    df_new,pca = gen_PCA(df)

    plt.bar(
    range(1,len(pca.explained_variance_)+1),
    pca.explained_variance_
    )


    plt.xlabel('PCA Feature')
    plt.ylabel('Explained variance')
    plt.title('Feature Explained Variance')
    plt.savefig('../figs/hashtags/pca.png')


    # LDA
    df = pd.read_csv('../data/BOW.csv')
    df_new,lda = gen_lda(df)

    plt.bar(
    range(1,len(lda.explained_variance_ratio_)+1),
    lda.explained_variance_ratio_
    )
    print(lda.coef_)

    plt.xlabel('lda Feature')
    plt.ylabel('Explained variance')
    plt.title('Feature Explained Variance')
    plt.savefig('../figs/hashtags/lda.png')
    '''
