# This script implements an auto encoder used to reduce the dimensionality of the vectorized text from 100 to 5

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses, Model
from random import randint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preproc_text import gen_dataframe

import os
import numpy as np 
import pandas as pd
from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.keras.utils import plot_model

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

if __name__ == "__main__":
    train_data = pd.read_csv("../../data/train.csv")
    # generate the bag of words
    df = gen_dataframe(train_data)
    print(df.shape)
    print(df)
    label = train_data["retweets_count"]

    X_train, x_test = train_test_split(df, test_size=0.1, random_state=45)
    print(X_train.shape)
    print(x_test.shape)

    # define the number of features
    ncol = X_train.shape[1]

    encoding_dim = 5

    input_dim = Input(shape = (ncol, ))

    # Encoder Layers
    encoded1 = Dense(128, activation = 'relu', name="Encoder-01")(input_dim)
    encoded2 = Dense(64, activation = 'relu', name="Encoder-02")(encoded1)
    encoded3 = Dense(32, activation = 'relu', name="Encoder-03")(encoded2)
    bottle_neck = Dense(encoding_dim, activation = 'relu', name="BottleNeck")(encoded3)

    # Decoder Layers
    decoded1 = Dense(64, activation = 'relu', name="Decoder-01")(bottle_neck)
    decoded2 = Dense(128, activation = 'relu', name="Decoder-02")(decoded1)
    decoded3 = Dense(256, activation = 'relu', name="Decoder-03")(decoded2)
    decoded4 = Dense(ncol, activation = 'sigmoid', name="Decoder-04")(decoded3)

    # Combine Encoder and Deocder layers
    autoencoder = Model(inputs = input_dim, outputs = decoded4)

    # Compile the Model
    autoencoder.compile(optimizer = 'adadelta', loss = losses.MeanSquaredError())

    history = autoencoder.fit(X_train, X_train, epochs = 50, batch_size = 64, shuffle = False, validation_data = (x_test, x_test), callbacks=[callback])

    # Use Encoder level to reduce dimension of train and test data¶
    encoder = Model(inputs = input_dim, outputs = bottle_neck)
    encoded_input = Input(shape = (encoding_dim, ))

    encoded_train = pd.DataFrame(encoder.predict(df))
    encoded_train = encoded_train.add_prefix('text_encoded_')

    encoded_train["retweets_count"] = label
    print(encoded_train)
    encoded_train.to_csv('text_encoded.csv', index=False)