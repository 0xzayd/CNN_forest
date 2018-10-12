# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 14:40:36 2018

@author: zaha
"""
import os
import gdal
import glob
from osgeo import gdal_array
import numpy as np
from src import CNN2DModel

def prediction(image, threshold):
    loaded_model = CNN2DModel(num_gpus = 1, sim_id = 1)

    loaded_model.build_model(blocks=[16,32,64,128])

    #myModel.compile_model(lr = 0.01, verbose=True)
    
    loaded_model.load_weights("C:/Users/zaha/Test_Keras/GPU_SEP/weights_final.hdf5")
    
    tif_path = image
    sheet = tif_path.split(os.path.sep)[-1][:-4]
    tif_to_mask(model=loaded_model, sheet=os.path.dirname(image) + os.path.sep + sheet, threshold=threshold, output=os.path.dirname(image) + os.path.sep + sheet+ 'thshld_' + str(threshold) + '_predicted.tif')
    
    
def ary_to_tiles(ary, shape=(256,256), exclude_empty=False):
    """
    Function to turn a big 2D numpy array (image) and tile it into a set number of shapes
    
    Outputs a stacked numpy array suitable for input into a Convolutional Neural Network
    """
    assert(isinstance(ary, np.ndarray))
    assert(isinstance(shape, tuple))
    
    ary_height, ary_width = shape
    ary_list = []
    
    total = 0
    excluded = 0
    for x_step in range(0, ary.shape[1], ary_width):
        for y_step in range(0, ary.shape[0], ary_height):
            x0, x1 = x_step, x_step+ary_width
            y0, y1 = y_step, y_step+ary_height
            crop_ary = ary[y0:y1, x0:x1]
            try:
                total += 1
                assert(crop_ary.shape == (ary_height, ary_width, ary.shape[2]))  #do not include images not matching the intended size
            except AssertionError:
                excluded += 1
                #print(y0,y1,x0,x1, 'excluded')
                continue
            ary_list.append(crop_ary)
    
    if excluded > 0:
        print("INFO: {0}/{1} tiles were excluded due to not fitting shape {2}".format(excluded, total, shape))
    return np.stack(ary_list), excluded


    
def tiles_to_ary(stacked_ary, final_shape=(256, 256)):
    """
    Function to turn a stacked 2D numpy array of shape (tiles, height, width, channels)
    into a single 2D numpy array (image) of shape (height, width, channels)
    
    Outputs a single numpy array suitable for converting into a raster such as a Geotiff
    """
    
    assert(len(stacked_ary.shape) == 4)
    
    ary_height = stacked_ary.shape[1]
    ary_width = stacked_ary.shape[2]
    
    output_ary = np.zeros(shape=final_shape+(stacked_ary.shape[3],))

    index = 0
    for x_step in range(0, final_shape[1], ary_width):
        for y_step in range(0, final_shape[0], ary_height):
            x0, x1 = x_step, x_step+ary_width
            y0, y1 = y_step, y_step+ary_height
            
            output_ary[y0:y1,x0:x1] = stacked_ary[index]
            index += 1
    
    return output_ary



def tif_to_mask(model, sheet:str, display:bool=False, threshold = 0.5, output:str=None):
    ds = gdal.Open('{0}.tif'.format(sheet)) #get from https://data.linz.govt.nz/layer/51870-wellington-03m-rural-aerial-photos-2012-2013/
    ary = np.dstack([ds.GetRasterBand(i).ReadAsArray() for i in range(1,5)])
    print('Input tif has shape:', ary.shape)
    
    # Convert raster array to tiles, need to ensure it is perfectly tiled!!
    W_test, excluded = ary_to_tiles(ary, shape=(256, 256))
    if excluded > 0:
        print((excluded,), *ary.shape)
        raise ValueError('''
        Need to ensure perfect tiles to create full mask of input tif! 
        You have missed {0} tiles from an input raster array of shape ({1},{2}) 
        Try and find common factors for the input shape {1},{2} 
        to input into the (img_height, img_width) parameters instead of ({4},{5})
        '''.format((excluded,), *ary.shape, 256, 256))

    W_hat_test = model.predict(W_test, verbose=1)
    #print('Finished predict on {0} tiles of shape ({1},{2}) for: {4}'.format(*W_hat_test.shape + (sheet,)))
    
    W_hat_ary = tiles_to_ary(stacked_ary=W_hat_test, final_shape=ary.shape[:2])
    
    output_ary = W_hat_ary[:,:,0]
    output_ary[output_ary<threshold] = 0
    output_ary[output_ary>=threshold] = 1
        
    if output != None:
        print('Output to:', output)
        out_ds = gdal_array.SaveArray(output_ary, output, "gtiff", prototype=ds)
        del out_ds
    
    del ds  #close opened tif
    
    return W_hat_ary

import argparse

parser = argparse.ArgumentParser(description='arguments input')
parser.add_argument('-p','--path', type=str, help='path to the test files', required=True)
args = parser.parse_args()
path = args.path

for _,r in enumerate(sorted(glob.glob(path + '/TEST*.tif'))):
    if not r.endswith('predicted.tif'):
        for i in range(10,100,5):
            prediction(r, 0.01*i)
