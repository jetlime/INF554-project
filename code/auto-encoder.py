import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses, Model
from random import randint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preproc_BOW import gen_BOW_dataframe


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


if __name__ == "__main__":
    train_data = pd.read_csv("../data/train.csv")
    # tmp = gen_BOW_data(train_data)
    # pprint(tmp[0][-10:])
    # pprint(tmp[1][-10:])
    # pprint(np.max(tmp[1]))
    df = gen_BOW_dataframe(train_data, None)
    print(df.shape)
    print(df)
    label = df["retweets_count"]
    df = df.drop(['retweets_count'], axis=1)
    X_train, x_test = train_test_split(df, test_size=0.1, random_state=45)
    print(X_train.shape)
    print(x_test.shape)
    '''
    hidden_size = 100
    latent_size = 2

    input_layer = layers.Input(shape = X_train.shape[1:])
    flattened = layers.Flatten()(input_layer)
    hidden = layers.Dense(hidden_size, activation = 'relu')(flattened)
    latent = layers.Dense(latent_size, activation = 'relu')(hidden)
    encoder = Model(inputs = input_layer, outputs = latent, name = 'encoder')
    encoder.summary()
        
    input_layer_decoder = layers.Input(shape = encoder.output.shape)
    upsampled = layers.Dense(hidden_size, activation = 'relu')(input_layer_decoder)
    upsampled = layers.Dense(encoder.layers[1].output_shape[-1], activation = 'relu')(upsampled)
    constructed = layers.Reshape(X_train.shape[1:])(upsampled)
    decoder = Model(inputs = input_layer_decoder, outputs = constructed, name= 'decoder')
    decoder.summary()
        
    autoencoder = Model(inputs = encoder.input, outputs = decoder(encoder.output))
    autoencoder.summary()
        
    
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
    history = autoencoder.fit(X_train, X_train, epochs=1, batch_size=150, validation_data = (x_test, x_test), callbacks=[callback])

    '''

    # define the number of features
    ncol = X_train.shape[1]

    encoding_dim = 3

    input_dim = Input(shape = (ncol, ))

    # Encoder Layers
    encoded7 = Dense(128, activation = 'relu')(input_dim)
    encoded11 = Dense(64, activation = 'relu')(encoded7)
    encoded12 = Dense(32, activation = 'relu')(encoded11)
    encoded13 = Dense(encoding_dim, activation = 'relu')(encoded12)

    # Decoder Layers
    decoded1 = Dense(64, activation = 'relu')(encoded13)
    decoded2 = Dense(128, activation = 'relu')(decoded1)
    decoded3 = Dense(256, activation = 'relu')(decoded2)
    decoded5 = Dense(ncol, activation = 'sigmoid')(decoded3)

    # Combine Encoder and Deocder layers
    autoencoder = Model(inputs = input_dim, outputs = decoded5)

    # Compile the Model
    autoencoder.compile(optimizer = 'adadelta', loss = losses.MeanSquaredError())


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
        
    plt.savefig('../figs/hashtags/auto-encoder.png')