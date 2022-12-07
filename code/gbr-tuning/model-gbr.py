##################################################
## A script to train to hypertune the GBR regressor 
##################################################

import csv
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from verstack.stratified_continuous_split import scsplit # pip install verstack
from sklearn.model_selection import GridSearchCV
from pipeline_gbr import featurePipeline

# finetuning parameters 
parameters = {'learning_rate': [0.01,0.02,0.03,0.04],
				'subsample'    : [0.9, 0.5, 0.2, 0.1],
				'n_estimators' : [100,500,1000],
				'max_depth'    : [4,6,8,10]
				}

if __name__ == "__main__":
	# Load the training data
	train_data = pd.read_csv("../../data/train.csv")

	pd.set_option('display.max_columns', 1000)

	# Here we split our training data into trainig and testing set. This way we can estimate the evaluation of our model without uploading to Kaggle and avoid overfitting over our evaluation dataset.
	# scsplit method is used in order to split our regression data in a stratisfied way and keep a similar distribution of retweet counts between the two sets
	X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweets_count'], stratify=train_data['retweets_count'], train_size=0.7, test_size=0.3)

	X_train, X_test = featurePipeline(X_train, X_test, True)

	# Now we can train our model. Here we chose a Gradient Boosting Regressor and we set our loss function 
	reg = GradientBoostingRegressor(random_state=0)
	clf = GridSearchCV(reg, parameters, verbose=3, scoring="neg_mean_absolute_error", cv=2, refit=True, return_train_score=True, n_jobs=-1, pre_dispatch='1.5*n_jobs')

	# We fit our model using the training data
	clf.fit(X_train, y_train)
	# Print the results of the grid search
	print(clf.cv_results_)
	print("The best parameters are, \n")
	print(clf.best_params_)