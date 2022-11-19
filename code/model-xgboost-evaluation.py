##################################################
## A script to train to evaluate the XGBoost
## regressor 
##################################################
## Author: Paul Houssel
## Last Updated: Nov 19 2022, 22:56
##################################################

import csv
import pickle
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from verstack.stratified_continuous_split import scsplit # pip install verstack
from nltk.corpus import stopwords 
from xgboost import XGBRegressor

from nltk import download
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

from pipeline import featurePipeline


if __name__ == "__main__":

	# Prediction on the evaluation dataset
	# Load the evaluation data
	eval_data = pd.read_csv("../data/evaluation.csv")
	# Pipe the evaluation data through the dataset
	tweetID = eval_data['TweetID']
	eval_data, _ = featurePipeline(eval_data, None, False)

	results = {}

	# loop through all model files to find the XGboost models
	for root, dirs, files	 in os.walk("../models"):
		for file in files:
			filename = os.path.join(root,file)
			parameter = filename.split("-")[1]
			iteration = filename.split("-")[2] 
			if(filename.split("-")[0]=="xgboost"):					
				with open(filename, 'rb') as f:
    				reg = pickle.load(f)
				# And then we predict the values for our testing set
				y_pred = reg.predict(eval_data)

				# We want to make sure that all predictions are non-negative integers
				y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
				