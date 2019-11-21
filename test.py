from keras.models import load_model
import matplotlib.pyplot as plt
import imageio
from skimage import io
import skimage
from tqdm import tqdm
from keras.models import model_from_json
from sklearn.metrics import classification_report
import numpy as np
import utils
import json
from glob import glob
import sys
from metrics import *

#load json and create model

if __name__ == '__main__':
    if len(sys.argv) == 4:
        model_name = sys.argv[1]
        patient_no = int(sys.argv[2])
    else:
        model_name = input('Model name: ')
        patient_no = int(input('Patient no: '))
#        slice_no = int(input('Slice no: '))

    json_file = open('outputs/models/tri_path_test.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)



    model.load_weights("outputs/models/{}_train.h5".format(model_name))
#print(model.summary())

    with open('config.json') as config_file:
        config = json.load(config_file)
    root = config['root']



    with open('config.json') as config_file:
        config = json.load(config_file)
    root = config['root']
    patient_path = glob(root + '/*pat{}*'.format(patient_no))
    patient_scans = utils.load_test_scans(patient_path[0])
    patient_scans = utils.norm_test_scans(patient_scans)

    gt = []
    pred = []


    pbar = tqdm(total = patient_scans.shape[0])
    for slice_no in range(patient_scans.shape[0]):
        test_slice = patient_scans[slice_no:slice_no+1,:,:,:4]
        test_label = patient_scans[slice_no:slice_no+1,:,:,4] 
        prediction = model.predict(test_slice, batch_size=256, verbose=0)
        prediction = prediction[0]
        prediction = np.around(prediction)
        prediction = np.argmax(prediction, axis=-1)
        truth = test_label[0,15:223,15:223]
#        truth = truth.reshape(43264,)
#        prediction = prediction.reshape(43264,)

        scan = test_slice[0,:,:,2]
        tmp_label = test_label[0]
        tmp_pred = prediction.reshape(208, 208)
        tmp_pred = np.pad(tmp_pred, (16,16), mode='edge')


        plt.figure(figsize=(15, 10))
        plt.subplot(131)
        plt.title('Input')
        plt.imshow(scan, cmap='gray')
        plt.subplot(132)
        plt.title('Ground Truth')
        plt.imshow(tmp_label,cmap='gray')
        plt.subplot(133)
        plt.title('Prediction')
        plt.imshow(tmp_pred,cmap='gray')
        plt.savefig('outputs/{}_pat{}_slice{}.png'.format(model_name,patient_no,slice_no), bbox_inches='tight')

        gt.extend(truth)
        pred.extend(prediction)
        pbar.update(1)


    pbar.close()
    gt = np.array(gt)
    pred = np.array(pred)

    dice_whole = DSC_whole(pred, gt)
    dice_en = DSC_en(pred, gt)
    dice_core = DSC_core(pred, gt)

    sen_whole = sensitivity_whole(pred, gt)
    sen_en = sensitivity_en(pred, gt)
    sen_core = sensitivity_core(pred, gt)

    spec_whole = specificity_whole(pred, gt)
    spec_en = specificity_en(pred, gt)
    spec_core = specificity_core(pred, gt)

    '''
    haus_whole = hausdorff_whole(pred, gt)
    haus_en = hausdorff_en(pred, gt)
    haus_core = hausdorff_core(pred, gt)
    '''

    print("=======================================")
    print("Dice whole tumor score: {:0.4f}".format(dice_whole)) 
    print("Dice enhancing tumor score: {:0.4f}".format(dice_en)) 
    print("Dice core tumor score: {:0.4f}".format(dice_core)) 
    print("=======================================")
    print("Sensitivity whole tumor score: {:0.4f}".format(sen_whole)) 
    print("Sensitivity enhancing tumor score: {:0.4f}".format(sen_en)) 
    print("Sensitivity core tumor score: {:0.4f}".format(sen_core)) 
    print("=======================================")
    print("Specificity whole tumor score: {:0.4f}".format(spec_whole)) 
    print("Specificity enhancing tumor score: {:0.4f}".format(spec_en)) 
    print("Specificity core tumor score: {:0.4f}".format(spec_core)) 
    '''
    print("=======================================")
    print("Hausdorff whole tumor score: {:0.4f}".format(haus_whole)) 
    print("Hausdorff enhancing tumor score: {:0.4f}".format(haus_en)) 
    print("Hausdorff core tumor score: {:0.4f}".format(haus_core)) 
    '''
    print("=======================================\n\n")









'''

    for slice_no in range(patient_scans.shape[0]):
        test_slice = patient_scans[slice_no:slice_no+1,:,:,:4]
        test_label = patient_scans[slice_no:slice_no+1,:,:,4]
    scan = test_slice[0,:,:,2]
    tmp_label = test_label[0]




# added batch_size of 256 (was none before)
    prediction = model.predict(test_slice, batch_size=256, verbose=1)
    prediction = np.around(prediction)
    print(prediction.shape)
    prediction = np.argmax(prediction, axis=-1)
    print(prediction.shape)
    pred = prediction[0]
    print(pred.shape)
    print(pred)
    tmp_pred = pred.reshape(208, 208)
    tmp_pred = np.pad(tmp_pred, (16, 16), mode='edge')


    plt.figure(figsize=(15, 10))
    plt.subplot(131)
    plt.title('Input')
    plt.imshow(scan, cmap='gray')
    plt.subplot(132)
    plt.title('Ground Truth')
    plt.imshow(tmp_label,cmap='gray')
    plt.subplot(133)
    plt.title('Prediction')
    plt.imshow(tmp_pred,cmap='gray')
    plt.savefig('outputs/{}_pat{}_slice{}.png'.format(model_name,patient_no,slice_no), bbox_inches='tight')
#plt.show()




    y = test_label[0,15:223,15:223]

    truth = y.reshape(43264,)
    guess = pred.reshape(43264,)
    print('slice no {}'.format(slice_no))

    print(classification_report(truth, guess, labels=[0,1,2,3,4]))
    '''











