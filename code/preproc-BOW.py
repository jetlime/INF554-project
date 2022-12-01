import pandas as pd
import numpy as np
from pprint import pprint

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

def gen_BOW2(data):
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

    return list(map(lambda x: x[0], list(sorted(BOW.items(), key=lambda item: item[1])[-100:])))

def sentence_BOW(data, BOW):
    hashtags = h.strip("[]").split(", ")
    return dict(filter(lambda i: i[0] in shortBOW, B.items()))

def sentence_BOW2(data, BOW):
    """returns bit encoding of hashtag presence"""
    return np.array([1 if i in data else 0 for i in BOW])
    

def gen_BOW_data(data):
    BOW = gen_BOW2(data)
    X = [sentence_BOW2(i.strip('[]').split(", "), BOW) for i in data['hashtags']]
    Y = np.floor(np.log(data["retweets_count"]+1))
    return X, Y 

if __name__ == "__main__":
    train_data = pd.read_csv("../data/train.csv")
    tmp = gen_BOW_data(train_data)
    pprint(tmp[0][-10:])
    pprint(tmp[1][-10:])
