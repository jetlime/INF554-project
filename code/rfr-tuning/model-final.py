##################################################
## A script to train and evaluate the baseline RFR regressor 
##################################################

import csv
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from verstack.stratified_continuous_split import scsplit # pip install verstack
from nltk.corpus import stopwords 
from pipeline_rfr import featurePipeline

mini = 10

if __name__ == "__main__":
	for i in range(1,10):
		# Load the training data
		train_data = pd.read_csv("../../data/train.csv")

		pd.set_option('display.max_columns', 1000)

		# Here we split our training data into trainig and testing set. This way we can estimate the evaluation of our model without uploading to Kaggle and avoid overfitting over our evaluation dataset.
		# scsplit method is used in order to split our regression data in a stratisfied way and keep a similar distribution of retweet counts between the two sets
		X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweets_count'], stratify=train_data['retweets_count'], train_size=0.7, test_size=0.3)

		X_train, X_test = featurePipeline(X_train, X_test, True)

		print(X_train)

		# Now we can train our model. Here we chose a Gradient Boosting Regressor and we set our loss function 
		reg = RandomForestRegressor(bootstrap=True, max_depth=11, max_features='auto', min_samples_split=10,n_estimators=250, n_jobs=-1)
		
		# We fit our model using the training data
		reg.fit(X_train, y_train)
		# And then we predict the values for our testing set
		y_pred = reg.predict(X_test)
		# We want to make sure that all predictions are non-negative integers
		y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
		result_pred = mean_absolute_error(y_true=y_test, y_pred=y_pred)
		print("Test Prediction error:", result_pred)

		# Prediction on the evaluation dataset
		# Load the evaluation data
		eval_data = pd.read_csv("../../data/evaluation.csv")
		# Pipe the evaluation data through the dataset
		tweetID = eval_data['TweetID']
		eval_data, _ = featurePipeline(eval_data, None, False)

		# And then we predict the values for our testing set
		y_pred = reg.predict(eval_data)
		mini = result_pred
		
		# We want to make sure that all predictions are non-negative integers
		y_pred = [int(value) if value >= 0 else 0 for value in y_pred]
		if result_pred<mini:
			print("Found better result, saving the prediction to csv file")
			# Dump the results into a file that follows the required Kaggle template
			with open("../../results/predictions-rfr-tuned.txt", 'w') as f:
				writer = csv.writer(f)
				writer.writerow(["TweetID", "retweets_count"])
				for index, prediction in enumerate(y_pred):
					writer.writerow([str(tweetID.iloc[index]) , str(int(prediction))])	