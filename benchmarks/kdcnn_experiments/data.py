from tensorflow import keras
import numpy as np

class getDataset():
    def __init__(self, name):
        self.name = name
    
    def get_data(self):
        if self.name == 'mnist':
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        x_train = x_train/255.0
        x_test = x_test/255.0
        return x_train, keras.utils.to_categorical(y_train), x_test, y_test
    
