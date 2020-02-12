from glob import glob
import json
import os
import utils
import numpy as np
import sys
from keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
import random
import keras.backend as K
from losses import *
from models import *
from keras.callbacks import  ModelCheckpoint,Callback,LearningRateScheduler


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[2] == 'help':
        print('python3 train.py [model_name] [start_patient] [end_patient] [batch_size] [epochs] [validation_split]')
    elif len(sys.argv) == 7:
        model_name = sys.argv[1]
        start_pat = sys.argv[2]
        end_pat = sys.argv[3]
        bs = int(sys.argv[4])
        eps = int(sys.argv[5])
        vs = float(sys.argv[6])
    else:
        model_name = input('Model name: ')
        start_pat = input('Start patient: ')
        end_pat = input('End patient: ')
        eps = int(input('Epochs: '))
        bs = int(input('Batch size: '))
        vs = float(input('Validation split: '))


    model = load_model('models/{}_train.h5'.format(model_name), 
            custom_objects={'dice_coef': dice_coef})

    with open('config.json') as config_file:
        config = json.load(config_file)
    root = config['root']

    patches = []
    labels = []
    
    pat_idx = int(start_pat)
    while pat_idx <= int(end_pat):
        print("Processing patient {}:".format(pat_idx))
        path = glob(root + '/*pat{}*'.format(pat_idx))
        scans = utils.load_scans(path[0])
        scans = utils.norm_scans(scans)
        balanced_x, balanced_y = utils.generate_balanced(scans)
        patches.extend(balanced_x)
        labels.extend(balanced_y)
        pat_idx = pat_idx + 1
    shuffle = list(zip(patches, labels))
    np.random.shuffle(shuffle)
    patches, labels = zip(*shuffle)
    patches = np.array(patches)
    labels = np.array(labels)
    y = np.zeros((labels.shape[0],1,1,5))
    for i in range(labels.shape[0]):
        y[i,:,:,labels[i]] = 1

    model.fit(patches, y, epochs = eps, batch_size = bs, validation_split = vs)
    model.save('models/{}_train.h5'.format(model_name))



