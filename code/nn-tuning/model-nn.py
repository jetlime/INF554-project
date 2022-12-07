##################################################
## A script to train an hypertune the RFR regressor 
##################################################
import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, make_scorer
from verstack.stratified_continuous_split import scsplit # pip install verstack
from sklearn.model_selection import GridSearchCV
from pipeline_nn import featurePipeline

parameters = {
	# Number of trees in the forest
    'n_estimators': [100, 150, 200, 250, 300],
	#'n_estimators': [100, 150]
	# maimum number of levels in each decision tree
    'max_depth': [int(x) for x in np.linspace(1, 110, num = 11)],
	# maimum number of features considered for splitting a node (auto: same as #features ; sqrt: sqrt(#features))
	'max_features': ['auto', 'sqrt'],
	# Method of selecting samples for training each tree
	'bootstrap' : [True, False],
	# The minimum number of samples required to split an internal node
	'min_samples_split' : [2, 5, 10]
}

if __name__ == "__main__":
	# Load the training data
	train_data = pd.read_csv("../../data/train.csv")

	pd.set_option('display.max_columns', 1000)

	# Here we split our training data into trainig and testing set. This way we can estimate the evaluation of our model without uploading to Kaggle and avoid overfitting over our evaluation dataset.
	# scsplit method is used in order to split our regression data in a stratisfied way and keep a similar distribution of retweet counts between the two sets
	X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweets_count'], stratify=train_data['retweets_count'], train_size=0.8, test_size=0.2)

	X_train, X_test = featurePipeline(X_train, X_test, True)
	X_train.to_csv('x_train.csv', index=False)
	X_test.to_csv('x_test.csv', index=False)

	#X_train = pd.read_csv("x_train.csv")
	#X_test = pd.read_csv("x_test.csv")

	print(X_train)

	# Now we can train our model. Here we chose a Gradient Boosting Regressor and we set our loss function 
	reg = RandomForestRegressor(random_state=0, n_jobs=-1)
	clf = GridSearchCV(reg, parameters, verbose=3, scoring="neg_mean_absolute_error", cv=2, refit=True, return_train_score=True)
	# We fit our model using the training data
	clf.fit(X_train, y_train)
	# Print the results of the grid search
	print(clf.cv_results_)
	print("The best parameters are, \n")
	print(clf.best_params_)