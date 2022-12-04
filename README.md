# INF 554 - Machine and Deep Learning Data Challenge: Retweet Prediction

### Authors: Paul Houssel, Silviu-Andrea Maftei and Elouan Gros

## Reproduce our Results
In this section we present  and explain how to execute our code in order to reproduce our obtained results. In order to make our results reproducable, we set a fixed random set and specify the exact versions of the used python libraries (see the file ```requirements.txt``` in the root folder).  
## Folders and Experimental scripts
In this section we explain all the code scripts that were use to experiment various hypothesis.
- **./code**
    - **pipeline.py**, A python method to preprocess the dataset, such that it can be used for the training and the evaluation of our models. 
    - **./hashtags-engineering**
    Containing all scripts for the encoding and dimensionality reduction of the hashtags. 
    - **./text-engineering**
    Containing all scripts for the vectorization and dimensionality reduction of the text contained in the tweet.
    - **correlation.py**
    A script to plot all the correlation plots and correlation matrices of the original and engineered features
    - **normalisation.py**
    A script to inspect the data distribution of certain features and the effect of given normalisation techniques. 
- **./Figs**
- **./models**
- **./results**