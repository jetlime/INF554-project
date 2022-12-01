import pandas as pd
from pprint import pprint

def gen_BOW(data):
    """generate big bag of words"""
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

if __name__ == "__main__":
    train_data = pd.read_csv("../data/train.csv")
    tmp = gen_BOW(train_data)
    pprint(tmp[0])
    pprint(tmp[1][-100:])
    
