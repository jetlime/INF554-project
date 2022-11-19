import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from verstack.stratified_continuous_split import scsplit # pip install verstack
from nltk.corpus import stopwords 
from nltk import download
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

def featurePipeline(X_train, X_test, test):
	if(test):
		# We remove the actual number of retweets from our features since it is the value that we are trying to predict
		X_test = X_test.drop(['retweets_count'], axis=1)
		X_train = X_train.drop(['retweets_count'], axis=1)

		X_test = X_test.drop(['timestamp'], axis=1)
		X_test = X_test.drop(['TweetID'], axis=1)
		X_test = X_test.drop(['friends_count'], axis=1)
		X_test["urls"] = np.where(X_test["urls"]=="[]",0 , 1)
		X_test["hashtags"] = np.where(X_test["hashtags"]=="[]",0 , 1)
		X_test["mentions"] = np.where(X_test["mentions"]=="[]",0 , 1)
		X_test = X_test.drop(['hashtags'], axis=1)
		X_test = X_test.drop(['urls'], axis=1)
		X_test = X_test.drop(['text'], axis=1)

		X_test = X_test.drop(['mentions'], axis=1)

		#X_test["statuses_count"] = X_test["statuses_count"].clip(upper = 100000)
		X_test["statuses_count"] = np.log(X_test["statuses_count"])
		# add one to ignore 0 cases, could be problematic for the log normalisation
		X_test["followers_count"] += 1
		X_test["followers_count"] = np.log(X_test["followers_count"])

	# We remove the actual number of retweets from our features since it is the value that we are trying to predict
	X_train = X_train.drop(['timestamp'], axis=1)
	X_train = X_train.drop(['TweetID'], axis=1)
	X_train = X_train.drop(['friends_count'], axis=1)
	X_train["urls"] = np.where(X_train["urls"]=="[]",0 , 1)
	X_train["mentions"] = np.where(X_train["mentions"]=="[]",0 , 1)
	X_train["hashtags"] = np.where(X_train["hashtags"]=="[]",0 , 1)		
	# Clipping Normalisation because ..., see normalisation-statues.png
	X_train["statuses_count"] = np.log(X_train["statuses_count"])
	X_train["followers_count"] += 1
	X_train["followers_count"] = np.log(X_train["followers_count"])
	X_train = X_train.drop(['hashtags'], axis=1)
	X_train = X_train.drop(['mentions'], axis=1)

	X_train = X_train.drop(['text'], axis=1)

	X_train = X_train.drop(['urls'], axis=1)

	return X_train, X_test

if __name__ == "__main__":
	# Load the training data
	train_data = pd.read_csv("../data/train.csv")

	pd.set_option('display.max_columns', 1000)

	# Here we split our training data into trainig and testing set. This way we can estimate the evaluation of our model without uploading to Kaggle and avoid overfitting over our evaluation dataset.
	# scsplit method is used in order to split our regression data in a stratisfied way and keep a similar distribution of retweet counts between the two sets
	X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweets_count'], stratify=train_data['retweets_count'], train_size=0.7, test_size=0.3)

	X_train, X_test = featurePipeline(X_train, X_test, True)

	print(X_train)

	# Now we can train our model. Here we chose a Gradient Boosting Regressor and we set our loss function 
	reg = GradientBoostingRegressor()
	
	# We fit our model using the training data
	reg.fit(X_train, y_train)
	# And then we predict the values for our testing set
	y_pred = reg.predict(X_test)
	# We want to make sure that all predictions are non-negative integers
	y_pred = [int(value) if value >= 0 else 0 for value in y_pred]

	print("Test Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))


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

	# Dump the results into a file that follows the required Kaggle template
	with open("gbr_predictions.txt", 'w') as f:
		writer = csv.writer(f)
		writer.writerow(["TweetID", "retweets_count"])
		for index, prediction in enumerate(y_pred):
			writer.writerow([str(tweetID.iloc[index]) , str(int(prediction))])	