##################################################
## Data preprocessing pipeline
##################################################
## Author: Paul Houssel
## Last Updated: Nov 19 2022, 21:24
##################################################
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 

def unixSec2TOD(time : int):
	"""convert time in unix seconds to time of day (hour)"""
	secInDay = 86400 * 1000
	secInHour = 3600 * 1000
	return (time % secInDay) // secInHour

def unixSec2DOW(time : int):
	"""convert time in unix seconds to day of week (hour)"""
	secInWeek = 604800 * 1000
	secInDay = 86400 * 1000
	return (time % secInWeek) // secInDay

def sigmoid(arr):
    return 1 / (1 + np.exp(-arr))

def featurePipeline(X_train, X_test, test, drop=True):
	if(test):
		if drop:
			# We remove the actual number of retweets from our features since it is the value that we are trying to predict
			X_test = X_test.drop(['retweets_count'], axis=1)
			X_train = X_train.drop(['retweets_count'], axis=1)

		#------------Experimentation------------#
	
		# test out clipping normalisation
		#X_test["statuses_count"] = X_test["statuses_count"].clip(upper = 100000)		

		# put the text polarity into bins
		#X_test["text_polarity"] = X_test["text_polarity"].apply(binPolarity)

		# Vectorize the raw text of the tweet
		#X_test["text"] = vectorizer.fit_transform(X_test['text'])

		# Proper feature encoding
		#X_test["urls"] = np.where(X_test["urls"]=="[]",0 , 1)
		#X_test["hashtags"] = np.where(X_test["hashtags"]=="[]",0 , 1)
		#X_test["mentions"] = np.where(X_test["mentions"]=="[]",0 , 1)

		#X_test["favorites_count"] += 1
		#X_test["favorites_count"] = np.log(X_test["favorites_count"])
		#------------Experimentation------------#

		# Compute and bin the polarity of the text
		X_test["text_polarity"] = X_test["text"].apply(lambda text: TextBlob(text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment[0])
		X_test["text_sentiment"] = X_test["text"].apply(lambda text: TextBlob(text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment[1])

		# Count the hashtags and url's
		X_test["url_count"] = X_test["urls"].apply(lambda x: len(x.strip('][').split(', ')))
		X_test["hash_count"] = X_test["hashtags"].apply(lambda x: len(x.strip("][").split(', ')))

		# replace timestamp with time of day
		X_test["TimeOfDay"] = unixSec2TOD(X_test['timestamp'])
		X_test["DayOfWeek"] = unixSec2DOW(X_test['timestamp'])

		# Log normalisation
		# add one to ignore 0 cases, could be problematic for the log normalisation
		X_test["followers_count"] += 1
		X_test["followers_count"] = np.log(X_test["followers_count"])

		# REMOVED FEATURES		
		X_test = X_test.drop(['mentions'], axis=1)
		X_test = X_test.drop(['text'], axis=1)
		X_test = X_test.drop(['verified'], axis=1)
		X_test = X_test.drop(['TweetID'], axis=1)
		X_test = X_test.drop(['friends_count'], axis=1)
		X_test = X_test.drop(['hashtags'], axis=1)
		X_test = X_test.drop(['urls'], axis=1)
		X_test = X_test.drop(['statuses_count'], axis=1)
		X_test = X_test.drop(['timestamp'], axis=1)
		X_test = X_test.drop(['TimeOfDay'], axis=1)
		#X_test = X_test.drop(['DayOfWeek'], axis=1)
		X_test = X_test.drop(['url_count'], axis=1)

	#------------Experimentation------------#
	#X_train["text_polarity"] = X_train["text_polarity"].apply(binPolarity)

	#X_train["text"] = vectorizer.transform(X_train['text'])

	#X_train["urls"] = np.where(X_train["urls"]=="[]",0 , 1)
	#X_train["mentions"] = np.where(X_train["mentions"]=="[]",0 , 1)
	#X_train["hashtags"] = np.where(X_train["hashtags"]=="[]",0 , 1)	

	#X_train["favorites_count"] += 1
	#X_train["favorites_count"] = np.log(X_train["favorites_count"])
	#X_train["favorites_count"] = round(X_train["favorites_count"], 3)
	#------------Experimentation------------#

	X_train["text_polarity"] = X_train["text"].apply(lambda text: TextBlob(text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment[0])
	X_train["text_sentiment"] = X_train["text"].apply(lambda text: TextBlob(text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer()).sentiment[1])


	X_train["url_count"] = X_train["urls"].apply(lambda x: len(x.strip("][").split(', ')))
	X_train["hash_count"] = X_train["hashtags"].apply(lambda x: len(x.strip("][").split(', ')))

	X_train["TimeOfDay"] = unixSec2TOD(X_train['timestamp'])
	X_train["DayOfWeek"] = unixSec2DOW(X_train['timestamp'])
		
	X_train["followers_count"] += 1
	X_train["followers_count"] = np.log(X_train["followers_count"])
	
	
	X_train = X_train.drop(['TweetID'], axis=1)
	X_train = X_train.drop(['friends_count'], axis=1)
	X_train = X_train.drop(['hashtags'], axis=1)
	X_train = X_train.drop(['mentions'], axis=1)
	X_train = X_train.drop(['text'], axis=1)
	X_train = X_train.drop(['urls'], axis=1)
	X_train = X_train.drop(['verified'], axis=1)
	X_train = X_train.drop(['statuses_count'], axis=1)
	X_train = X_train.drop(['timestamp'], axis=1)
	X_train = X_train.drop(['TimeOfDay'], axis=1)
	#X_train = X_train.drop(['DayOfWeek'], axis=1)
	#X_train = X_train.drop(['hash_count'], axis=1)
	X_train = X_train.drop(['url_count'], axis=1)

	return X_train, X_test
