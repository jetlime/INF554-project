##################################################
## A script to train the final model
##################################################
## Author: Paul Houssel
## Last Updated: Nov 19 2022, 21:23
##################################################

import csv
import pandas as pd
import statistics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from verstack.stratified_continuous_split import scsplit # pip install verstack
from nltk.corpus import stopwords 

from nltk import download
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

from pipeline import featurePipeline

pd.set_option('display.max_columns', 1000)


if __name__ == "__main__":
	results = []
	min = 1000
	print("Testing baseline model with following features, \n")
	train_data = pd.read_csv("../data/train.csv")

	X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweets_count'], stratify=train_data['retweets_count'], train_size=0.7, test_size=0.3)

	X_train, X_test = featurePipeline(X_train, X_test, True)

	print(X_train.columns)

	for i in range(0,10):
		# Load the training data
		train_data = pd.read_csv("../data/train.csv")

		# Here we split our training data into trainig and testing set. This way we can estimate the evaluation of our model without uploading to Kaggle and avoid overfitting over our evaluation dataset.
		# scsplit method is used in order to split our regression data in a stratisfied way and keep a similar distribution of retweet counts between the two sets
		X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweets_count'], stratify=train_data['retweets_count'], train_size=0.7, test_size=0.3)

		X_train, X_test = featurePipeline(X_train, X_test, True)

		# Now we can train our model. Here we chose a Gradient Boosting Regressor and we set our loss function 
		reg = GradientBoostingRegressor()
		
		# We fit our model using the training data
		reg.fit(X_train, y_train)
		# And then we predict the values for our testing set
		y_pred = reg.predict(X_test)
		# We want to make sure that all predictions are non-negative integers
		y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
		res = mean_absolute_error(y_true=y_test, y_pred=y_pred)
		results.append(res)
		print("Test Prediction error for test number " + str(i) + " is " + str(results[i]) + " \n")

		# Prediction on the evaluation dataset
		# Load the evaluation data
		eval_data = pd.read_csv("../data/evaluation.csv")
		# Pipe the evaluation data through the dataset
		tweetID = eval_data['TweetID']
		eval_data, _ = featurePipeline(eval_data, None, False)

		# And then we predict the values for our testing set
		y_pred = reg.predict(eval_data)

		# We want to make sure that all predictions are non-negative integers
		y_pred = [int(value) if value >= 0 else 0 for value in y_pred]

		# Only save the best result
		if res<min:
			min = res 
			#  Dump the results into a file that follows the required Kaggle template
			print("...Saving the currently best result...")
			with open("../results/gbr_predictions.txt", 'w') as f:
				writer = csv.writer(f)
				writer.writerow(["TweetID", "retweets_count"])
				for index, prediction in enumerate(y_pred):
					writer.writerow([str(tweetID.iloc[index]) , str(int(prediction))])


		

	print("The mean average of the test prediciton is {}".format(str(statistics.mean(results))))
	