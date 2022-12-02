import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses, Model
from random import randint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preproc_BOW import gen_BOW_dataframe

if __name__ == "__main__":
    train_data = pd.read_csv("../data/train.csv")
    # tmp = gen_BOW_data(train_data)
    # pprint(tmp[0][-10:])
    # pprint(tmp[1][-10:])
    # pprint(np.max(tmp[1]))
    df = gen_BOW_dataframe(train_data, None)
    print(df.shape)
        
    X_train, x_test = train_test_split(df, test_size=0.2, random_state=45)
    print(X_train.shape)
    print(x_test.shape)

    hidden_size = 100
    latent_size = 3

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
    history = autoencoder.fit(X_train, X_train, epochs=5, batch_size=100, validation_data = (x_test, x_test))


    fig, axs = plt.subplots(figsize=(15,15))

    axs.plot(history.history['loss'])
    axs.plot(history.history['val_loss'])
    axs.title.set_text('Training Loss vs Validation Loss')
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Loss')
    axs.legend(['Train','Val'])
        
    plt.savefig('../figs/hashtags/auto-encoder.png')