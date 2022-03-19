from tensorflow import keras

class getCNN():
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.compile_kwargs = {
            "loss": "categorical_crossentropy", 
            "optimizer": keras.optimizers.Adam(1e-3)
            }

    def LeNet(self):
        network = keras.Sequential()

        # conv blocks
        network.add(keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', input_shape=self.input_shape))
        network.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        network.add(keras.layers.Activation('relu'))

        network.add(keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same'))
        network.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        network.add(keras.layers.Activation('relu'))

        network.add(keras.layers.Flatten())

        # fully-connected layers
        network.add(keras.layers.Dense(20, activation='relu'))
        network.add(keras.layers.Dense(20, activation='relu'))
        network.add(keras.layers.Dense(10, activation='softmax'))
        
        network.compile(**self.compile_kwargs)

        return network
