# INF 554 - Machine and Deep Learning Data Challenge: Retweet Prediction

### Authors: Paul Houssel, Silviu-Andrea Maftei and Elouan Gros

## Reproduce our Results
In this section we present  and explain how to execute our code in order to reproduce our obtained results. In order to make our results reproducable, we set a fixed random set and specify the exact versions of the used python libraries (see the file ```requirements.txt``` in the root folder).  
## Folders and Experimental scripts
In this section we explain all the code scripts that were use to experiment various hypothesis.
- **./code**
    - ***pipeline.py***
    A python method to preprocess the dataset, such that it can be used for the training and the evaluation of our final model 
    - ***final-model.py*** 
    Our best performing model
    - **./hashtags-engineering**
    Containing all scripts for the encoding and dimensionality reduction of the hashtags. 
    - **./text-engineering**
    Containing all scripts for the vectorization and dimensionality reduction of the text contained in the tweet.
    - **./baseline-models**
    Evaluation and training of all baseline models reported in section *3.2* of our report. For each model, there exists a given script to train the given model with the default model parameters after applying our feature engineering pipeline to the training and test set. 
    - **./grb-tuning**
    Hypertuning the Gradient Boosting regressor including the according data pipeline.
    - **./rfr-tuning**
    Hypertuning the Random Forest regressor including the according data pipeline.
    - **./xgb-tuning**
    Hypertuning the XG Boosting regressor including the according data pipeline.
    - **./data-analysis**
    All scripts that were used to generate figures and help us understand the data. 
        - ***time-vis.py***
        A data visualisation script investigating the results of transforming and combining UNIX time data
        - ***correlation.py***
        A script to plot all the correlation plots and correlation matrices of the original and engineered features
        - ***normalisation.py***
        A script to inspect the data distribution of certain features and the effect of given normalisation techniques. 
- **./figs**
- **./models**
- **./results**