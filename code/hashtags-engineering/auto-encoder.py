# This script implements an auto encoder used to reduce the dimensionality of the one-hot encoded hashtag Bag of Words features

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses, Model
from random import randint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preproc_BOW import gen_BOW_dataframe

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
    df = gen_BOW_dataframe(train_data, None)
    print(df.shape)
    print(df)
    label = df["retweets_count"]
    df = df.drop(['retweets_count'], axis=1)
    X_train, x_test = train_test_split(df, test_size=0.1, random_state=45)
    print(X_train.shape)
    print(x_test.shape)


    # define the number of features
    ncol = X_train.shape[1]

    encoding_dim = 3

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
    autoencoder = Model(inputs = input_dim, outputs = encoded3)

    # Compile the Model
    autoencoder.compile(optimizer = 'adadelta', loss = losses.MeanSquaredError())

    # plot and save the architecture of the model as an image file
    plot_model(autoencoder, to_file='auto-encoder.png', show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="LR",
    expand_nested=False,
n    layer_range=None)

    history = autoencoder.fit(X_train, X_train, epochs = 100, batch_size = 64, shuffle = False, validation_data = (x_test, x_test), callbacks=[callback])


    # Use Encoder level to reduce dimension of train and test data¶
    encoder = Model(inputs = input_dim, outputs = encoded13)
    encoded_input = Input(shape = (encoding_dim, ))

    encoded_train = pd.DataFrame(encoder.predict(df))
    encoded_train = encoded_train.add_prefix('hastag_encoded_')


    encoded_train["retweets_count"] = label


    encoded_train.to_csv('hashtahs_encoded.csv', index=False)



    fig, axs = plt.subplots(figsize=(15,15))

    axs.plot(history.history['loss'])
    axs.plot(history.history['val_loss'])
    axs.title.set_text('Training Loss vs Validation Loss')
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Loss')
    axs.legend(['Train','Val'])
        
    plt.savefig('../../figs/hashtags/auto-encoder.png')