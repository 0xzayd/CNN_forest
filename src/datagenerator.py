import keras
import os
import numpy as np

class DataGenerator(keras.utils.Sequence):

    #Generates data for Keras

    def __init__(self, pathX = './train/X_data.npz', pathY = './train/Y_data.npz', batch_size=32):

        if os.path.exists(pathX) and os.path.exists(pathY):
            self.X_data = np.load(pathX)['X_data']
            self.Y_data = np.load(pathY)['Y_data']
        else:
            raise ValueError("Path doesn't exist")

        self.batch_size = batch_size


    def __len__(self):

        #Denotes the number of batches per epoch
        return int(np.floor(len(self.X_data) / self.batch_size))

    def __getitem__(self, index):

        #Generate one batch of data

        # Generate data
        x = self.X_data[index*self.batch_size:(index+1)*self.batch_size]
        y = self.Y_data[index*self.batch_size:(index+1)*self.batch_size]

        return x, y