import os
import numpy as np
class Data:
    X_data = None
    Y_data = None
    def load_XY(self, pathX = './train/X_data.npz', pathY = './train/Y_data.npz'):
        if os.path.exists(pathX) and os.path.exists(pathY):
            X_data = np.load(pathX)['X_data']
            Y_data = np.load(pathY)['Y_data']
        return X_data, Y_data
