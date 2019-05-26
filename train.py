from glob import glob
import json
import os
import utils
import numpy as np
import sys
from keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight

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


if __name__ == '__main__':

    if len(sys.argv) == 3:
        model_name = sys.argv[1]
        patient_no = sys.argv[2]
    else:
        model_name = input('Model name: ')
        patient_no = input('Patient no: ')

    model = load_model('outputs/models/{}_train.h5'.format(model_name))

    with open('config.json') as config_file:
        config = json.load(config_file)
    root = config['root']
    
    patient_path = glob(root + '/*pat{}*'.format(patient_no))
    patient_scans = utils.load_scans(patient_path[0])
    patient_scans = utils.norm_scans(patient_scans)
    
    for slice_no, patient_slice in enumerate(patient_scans):
        gt = patient_slice[:,:,4]
        # exclude slices without any tumor present
        if len(np.argwhere(gt == 0)) == (240 * 240):
            continue
        x, labels = utils.training_patches(patient_slice)
        class_weights = compute_class_weight('balanced',np.unique(label),label)
        # reshape labels to (1,1,5) with one hot encoding
        y = np.zeros((labels.shape[0],1,1,5))
        for i in range(labels.shape[0]):
            y[i,:,:,labels[i]] = 1

        print('Slice no {}'.format(slice_no))
        model.fit(x,y,epochs=5,batch_size=128,class_weight=cw)
        model.save('outputs/models/{}_train.h5'.format(model_name))
        model.save_weights('{}_train_weights.h5'.format(model_name))





    
    

    















