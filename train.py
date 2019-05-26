from glob import glob
import json
import os
import utils
import numpy as np
import sys
from keras.models import load_model

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

    model = load_model('outputs/{}.h5'.format(model_name))

    with open('config.json') as config_file:
        config = json.load(config_file)
    root = config['root']
    
    patient_path = glob(root + '/*pat{}*'.format(patient_no))
    patient_scans = utils.load_scans(patient_path[0])
    patient_scans = utils.norm_scans(patient_scans)
    
    for slice_no, patient_slice in enumerate(patient_scans):
        gt = patient_slice[:,:,4]
        # exclude slices without a tumor present
        if len(np.argwhere(gt == 0)) != (240 * 240):
            x, y, class_weights = utils.training_patches(patient_slice)
            print('Slice no {}'.format(slice_no))
            model.fit(x,y,epochs=5,batch_size=128,class_weight=class_weights)
            model.save('outputs/models/{}.h5'.format(model_name))




    
    

    















