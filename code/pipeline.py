##################################################
## Data preprocessing pipeline
##################################################
## Author: Paul Houssel
## Last Updated: Nov 19 2022, 21:24
##################################################

import numpy as np

def unixSec2TOD(time : int):
	"""convert time in unix seconds to time of day (hour)"""
	secInDay = 86400
	secInHour = 3600
	return (time % secInDay) // secInHour

def unixSec2DOW(time : int):
	"""convert time in unix seconds to day of week (hour)"""
	secInWeek = 604800
	secInDay = 86400
	return (time % secInWeek) // secInDay

def featurePipeline(X_train, X_test, test, drop=True):
	if(test):
		# keeping it is sometimes good for data analysis
		if drop:
			# We remove the actual number of retweets from our features since it is the value that we are trying to predict
			X_test = X_test.drop(['retweets_count'], axis=1)
			X_train = X_train.drop(['retweets_count'], axis=1)

		# replace timestamp with time of day
		X_test["TimeOfDay"] = unixSec2TOD(X_test['timestamp'])
		X_test["DayOfWeek"] = unixSec2DOW(X_test['timestamp'])
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
	X_train["TimeOfDay"] = unixSec2TOD(X_train['timestamp'])
	X_train["DayOfWeek"] = unixSec2DOW(X_train['timestamp'])
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
