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

'''
with open('config.json') as config_file:
    config = json.load(config_file)

root = config['root']
pats = os.listdir(root)
pats = [os.path.join(root, name) for name in pats if 'pat' in name.lower()]

scans = utils.load_scans(pats[pat_no])

print('scans shape: {}'.format(scans.shape))
scans = utils.norm_scans(scans)
tmp_slice = scans[80]
if len(np.argwhere(tmp_slice[:,:,4] == 0)) != (240*240):
    patches, labels = utils.training_patches(tmp_slice)
    print('patches shape: {}'.format(patches.shape))
    print('labels shape: {}'.format(labels.shape))
else:
    print('zeros')
'''
def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_true, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0
    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score





if __name__ == '__main__':

    if len(sys.argv) == 2:
        model_name = sys.argv[1]
    else:
        model_name = input('Model name: ')

    model = load_model('outputs/models/{}_train.h5'.format(model_name), custom_objects={'f1_score': f1_score})

    with open('config.json') as config_file:
        config = json.load(config_file)
    root = config['root']
    
    for patient_no in range(190):

        patient_path = glob(root + '/*pat{}*'.format(patient_no))
        patient_scans = utils.load_scans(patient_path[0])
        patient_scans = utils.norm_scans(patient_scans)

        slice_no = random.randint(1,146)
        patient_slice = None 
        slice_found = False
        while slice_found == False:
            tmp_slice = patient_scans[slice_no,:,:,:]
            gt = patient_scans[slice_no,:,:,4]
            if len(np.argwhere(gt==0)) >= (192*152):
                continue
            else:
                patient_slice = tmp_slice
                slice_found = True

        x, labels = utils.training_patches(patient_slice)
        class_weights = compute_class_weight('balanced',np.unique(labels),labels)
        y = np.zeros((labels.shape[0],1,1,5))
        for i in range(labels.shape[0]):
             y[i,:,:,labels[i]] = 1

        
        print('Patient {} - Slice {}'.format(patient_no, slice_no))
        model.fit(x,y,epochs=3,batch_size=1024,class_weight=class_weights, validation_split=0.4)
        model.save('outputs/models/{}_train.h5'.format(model_name))
        model.save_weights('{}_train_weights.h5'.format(model_name))
     
        
        
        '''

        for slice_no, patient_slice in enumerate(patient_scans):
            gt = patient_slice[:,:,4]
            # exclude slices without any tumor present
            #if gt.all() == 0:
            #if len(np.argwhere(gt == 0)) == (240 * 240):
            # check if more than 97% of slice is 0
            #check if more than 96% of slice is 0
            if len(np.argwhere(gt==0)) == (192*152):
                continue
            x, labels = utils.training_patches(patient_slice)
            class_weights = compute_class_weight('balanced',np.unique(labels),labels)
            # reshape labels to (1,1,5) with one hot encoding
            y = np.zeros((labels.shape[0],1,1,5))
            for i in range(labels.shape[0]):
                y[i,:,:,labels[i]] = 1

            print('Slice no {}'.format(slice_no))
            # changed batch size to 256 from 1024 
            model.fit(x,y,epochs=3,batch_size=1024,class_weight=class_weights, validation_split=0.4)
            model.save('outputs/models/{}_train.h5'.format(model_name))
            model.save_weights('{}_train_weights.h5'.format(model_name))

        '''




    
    

    















