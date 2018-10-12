# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 18:00:43 2018

@author: zaha
"""
import glob
import os
import argparse
from confusion import confusion

splitName = lambda filename: filename.split(os.sep)[-1].split('_predicted')[0] #function to strip filename of its directories and extension
splitName2 = lambda filename: filename.split(os.sep)[-1].split('thshld_')[-1] #function to strip filename of its directories and extension

parser = argparse.ArgumentParser(description='arguments input')
parser.add_argument('-gt','--tpath', type=str, help='path to the GT file', required=False)
parser.add_argument('-p','--ppath', type=str, help='path to prediction files', required=True)
args = parser.parse_args()
truth = args.tpath
pred = args.ppath

if truth == None:
    truth = 'C:/Users/zaha/Test_Keras/GPU_SEP/predict/groundT2/GroundTruth_test2.tif'
list_points = []

for _,rpred in enumerate(sorted(glob.glob(pred+'/*thshld_*_predicted.tif'))):
    name = splitName(os.path.basename(rpred))
    threshold = splitName2(name)
    
    tn,fp,fn,tp = confusion(truth, rpred)
    
    iou = tp/(tp+fp+fn)
    acc = (tn+tp)/(tn+tp+fn+fp)
    list_points.append({'threshold': threshold,'iou': iou,'acc': acc})

print(list_points)
