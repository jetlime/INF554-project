##################################################
## A script to train and evaluate the baseline NN regressor 
##################################################

import csv
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from verstack.stratified_continuous_split import scsplit # pip install verstack
from nltk.corpus import stopwords 
from pipeline_nn import featurePipeline

import math
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError

hidden_units1 = 160
hidden_units2 = 480
hidden_units3 = 256
learning_rate = 0.01

def build_model_using_sequential():
  model = Sequential([
    Dense(hidden_units1, kernel_initializer='normal', activation='relu'),
    Dropout(0.2),
    Dense(hidden_units2, kernel_initializer='normal', activation='relu'),
    Dropout(0.2),
    Dense(hidden_units3, kernel_initializer='normal', activation='relu'),
    Dense(1, kernel_initializer='normal', activation='linear')
  ])
  return model


def plot_history(history, key):
  plt.plot(history.history[key])
  plt.plot(history.history['val_'+key])
  plt.xlabel("Epochs")
  plt.ylabel(key)
  plt.legend([key, 'val_'+key])
  plt.show()

if __name__ == "__main__":
	# Load the training data
	train_data = pd.read_csv("../../data/train.csv")

	pd.set_option('display.max_columns', 1000)

	# Here we split our training data into trainig and testing set. This way we can estimate the evaluation of our model without uploading to Kaggle and avoid overfitting over our evaluation dataset.
	# scsplit method is used in order to split our regression data in a stratisfied way and keep a similar distribution of retweet counts between the two sets
	X_train, X_test, y_train, y_test = scsplit(train_data, train_data['retweets_count'], stratify=train_data['retweets_count'], train_size=0.7, test_size=0.3)

	X_train, X_test = featurePipeline(X_train, X_test, True)

	X_test = tf.convert_to_tensor(X_test)
	X_train = tf.convert_to_tensor(X_train)

	print(X_train)

	# build the model
	model = build_model_using_sequential()

	# loss function
	mse = mean_absolute_error
	model.compile(
		loss=mse, 
		optimizer=Adam(learning_rate=learning_rate), 
		metrics=[mse]
	)
	with tf.device("/GPU:0"):
		# train the model
		history = model.fit(
			X_train.values, 
			y_train.values, 
			epochs=10, 
			batch_size=64,
			validation_split=0.2, verbose=2
		)

	y_pred = model.predict(X_test)
	# We want to make sure that all predictions are non-negative integers
	y_pred = [int(value) if value >= 0 else 0 for value in y_pred]

	print("Test Prediction error:", mean_absolute_error(y_true=y_test, y_pred=y_pred))


	# Prediction on the evaluation dataset
	# Load the evaluation data
	eval_data = pd.read_csv("../../data/evaluation.csv")
	# Pipe the evaluation data through the dataset
	tweetID = eval_data['TweetID']
	eval_data, _ = featurePipeline(eval_data, None, False)

	# And then we predict the values for our testing set
	y_pred = reg.predict(eval_data)

	# We want to make sure that all predictions are non-negative integers
	y_pred = [int(value) if value >= 0 else 0 for value in y_pred]

	# Dump the results into a file that follows the required Kaggle template
	with open("../../results/predictions-rfr.txt", 'w') as f:
		writer = csv.writer(f)
		writer.writerow(["TweetID", "retweets_count"])
		for index, prediction in enumerate(y_pred):
			writer.writerow([str(tweetID.iloc[index]) , str(int(prediction))])	