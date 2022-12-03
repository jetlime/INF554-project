##################################################
## A script to train to hypertune the XGBoost
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


# Hypertuning parameters
boosters = ["gbtree", "dart"] #"gblinear", "dart"]

if __name__ == "__main__":
	# Load the training data
	train_data = pd.read_csv("../data/train.csv")

	pd.set_option('display.max_columns', 1000)

	# Here we split our training data into trainig and testing set. This way we can estimate the evaluation of our model without uploading to Kaggle and avoid overfitting over our evaluation dataset.
	# scsplit method is used in order to split our regression data in a stratisfied way and keep a similar distribution of retweet counts between the two sets
	X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweets_count'], stratify=train_data['retweets_count'], train_size=0.7, test_size=0.3)

	X_train, X_test = featurePipeline(X_train, X_test, True)
	for booster in boosters :
		# Now we can train our model. Here we chose a Gradient Boosting Regressor and we set our loss function 
		#reg = XGBRegressor(booster=booster, verbosity=1)
		reg = XGBRegressor(booster=booster, verbosity=1,\
			tree_method="gpu_hist", gpu_id=0,\
			learning_rate=0.01,\
			#n_estimators=100,\
			max_depth=5,\
			gamma=1)
	
		# We pre-fit our model using the training data
		reg.fit(X_train, y_train)

		for i in range(1,10):
			#print("Fitting model using {}, iteration {}\n".format(booster, str(i)))

			# We fit our model using the training data
			reg.fit(X_train, y_train,\
                    xgb_model=reg)

			# Save the model as a pickle file
			with open("../models/xgboost-{}-{}".format(booster, str(i)), 'wb') as file:
				pickle.dump(reg, file)

			y_ctrl = reg.predict(X_train)
			# And then we predict the values for our testing set
			y_pred = reg.predict(X_test)
			# We want to make sure that all predictions are non-negative integers
			y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
			#print("Test Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))
			print('{}:{} -> Train: {} \t Test: {}'.format(
				booster, i,
				mean_absolute_error(y_true=y_train, y_pred=y_ctrl),
				mean_absolute_error(y_true=y_test, y_pred=y_pred)
				))
