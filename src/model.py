import keras
import re
import glob
from .trainvaltensorboard import TrainValTensorBoard
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, AlphaDropout
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import multi_gpu_model
from .utils import mean_iou

class CNN2DModel:
    
    def __init__(self, num_gpus = 1,  sim_id = 1):
        self.model = None
        self.callbackList = None
        self.initial_epoch = 0
        self.num_gpus = num_gpus
        self.sim_id = sim_id
        self.path = './model'+str(sim_id)
        return
    
    def build_model(self, img_height=256, img_width=256, activation = 'selu', filter_initializer = 'lecun_normal', blocks = [64, 64, 64, 64] ,use_tfboard = False):
        
        bands = 4
        if K.image_data_format() == 'channels_first':
            ch_axis = 1
            input_shape = (bands, img_height, img_width)
        if K.image_data_format() == 'channels_last':
            ch_axis = 3
            input_shape = (img_height, img_width, bands)
            
        inp = Input(input_shape)
        encoder = inp
        list_encoders = []
        
        print('building 2D Convolution Model ...')
        print(blocks)
        
        # Setting up the filter size
        filter_size = (3,3)
        # Encoding
        for block_id , n_block in enumerate(blocks):
            with K.name_scope('Encoder_block_{0}'.format(block_id)):
                encoder = Conv2D(filters = n_block, kernel_size = filter_size, activation = activation, padding = 'same',
                                kernel_initializer = filter_initializer)(encoder)
                encoder = AlphaDropout(0,1*block_id, )(encoder)
                encoder = Conv2D(filters = n_block, kernel_size = filter_size, dilation_rate = (2,2),
                                 activation = activation, padding='same', kernel_initializer = filter_initializer)(encoder)
                
                list_encoders.append(encoder)
                # maxpooling 'BETWEEN' every 2 blocks
                if block_id < len(blocks)-1:
                    encoder = MaxPooling2D(pool_size = (2,2))(encoder)

        # Decoding
        decoder = encoder
        decoder_blocks = blocks[::-1][1:]
        for block_id, n_block in enumerate(decoder_blocks):
            with K.name_scope('Decoder_block_{0}'.format(block_id)):
                block_id_inv = len(blocks) - 1 - block_id
                decoder = concatenate([decoder, list_encoders[block_id_inv]], axis = ch_axis) # concatenate the first decoder with the last encoder and so on, according to the channell axis
                decoder = Conv2D(filters=n_block, kernel_size = filter_size, activation = activation, padding = 'same',
                                dilation_rate = (2,2), kernel_initializer = filter_initializer)(decoder)
                decoder = AlphaDropout(0,1*block_id, )(decoder)
                decoder = Conv2D(filters=n_block, kernel_size = filter_size, activation = activation, padding = 'same',
                                kernel_initializer = filter_initializer)(decoder)
                decoder = Conv2DTranspose(filters=n_block, kernel_size = filter_size, kernel_initializer = filter_initializer,
                                         padding='same', strides=(2,2))(decoder)
                
        # Last Layer...
        outp = Conv2DTranspose(filters=1, kernel_size = filter_size, activation = 'sigmoid',
                               padding = 'same', kernel_initializer = 'glorot_normal')(decoder)
            
        self.model = Model(inputs=[inp], outputs=[outp])
        
        return
    
    def get_weights(self, path_to_weights):
        return self.model.get_weights(path_to_weights)
    
    def set_weights(self, weights):
        self.model.set_weights(weights)
        return

    def compile_model(self, loss = 'binary_crossentropy', lr = 1e-3, verbose = True):
        
        print('compiling the model ...')
        # number of GPUs
        num_gpus = self.num_gpus
        # optimizer and metrics
        optimizer = keras.optimizers.adam(lr=lr)
        metrics = [mean_iou] 

        # For more GPUs
        if num_gpus > 1:
            self.model = multi_gpu_model(self.model, gpus=num_gpus)

        # Building the computational graph
        self.model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

        # print parameters of each layer
        if verbose:
            print(self.model.summary())
            
        # save the model template
        model_json = self.model.to_json()
        with open(self.path + "/model.json", "w") as json_file:
            json_file.write(model_json)
        return
    
    def build_callbacks(self, log_dir = None):

        # Tensorboard
        tensorboard = TrainValTensorBoard(log_dir = log_dir)
        
        # Model Checkpoints
        filepath=self.path + "/weights-{epoch:02d}-{val_mean_iou:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_mean_iou', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
        
        # Bring all the callbacks together into a python list
        self.callbackList = [tensorboard, checkpoint]
        
    
    def load_checkpoint(self):
      
        try:
            checkfile = sorted(glob.glob(self.path + '/weights-*-*.hdf5'))[-1]
            self.model.load_weights(checkfile)
            self.initial_epoch = int(re.search(r"weights-(\d*)-", checkfile).group(1))
            print('weights loaded, resuming from epoch {0}'.format(self.initial_epoch))
        except IndexError:
            try:
                self.model.load_weights(self.path+'/model-weights.hdf5')
                print('weights loaded, starting from epoch {0}'.format(self.initial_epoch))
            except OSError:
                pass
        return
    
    def fit_model(self, X_train, Y_train, verbose = 1, validation_split=0.2, batch_size=32, epochs=120):
        self.model.fit(X_train,Y_train, verbose=verbose, validation_split=validation_split,
                      batch_size=batch_size, epochs=epochs, callbacks=self.callbackList,
                      initial_epoch=self.initial_epoch)
        return
    
    #def fit_model_generator(self, training_generator, validation_generator, epochs=120, verbose=1):
        #self.model.fit_generator(generator=training_generator,
                    #validation_data=validation_generator,
                    #verbose=verbose,
                    #epochs=epochs,
                    #initial_epoch=self.initial_epoch,
                    #callbacks=self.callbackList,
                    #use_multiprocessing=True,
                    #workers=6)  
    
    def save_weights(self):
        if self.num_gpus > 1:
            model_out = self.model.layers[-2]
        else:   
            model_out = self.model
        model_out.save_weights(filepath=self.path+"/model-weights.hdf5")
        return 
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath=filepath)
        return
    
    def savetheweightfile(self, weightsfilepath):
        self.load_weights(weightsfilepath)
        self.save_weights()
    
    def predict(self, X_test, verbose=1):
        return self.model.predict(X_test, verbose = verbose)
    
