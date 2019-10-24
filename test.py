from keras.models import load_model
import matplotlib.pyplot as plt
import imageio
from skimage import io
import skimage

from keras.models import model_from_json
from sklearn.metrics import classification_report
import numpy as np
import utils
import json
from glob import glob

#load json and create model

json_file = open('outputs/models/tri_path_test.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights("outputs/models/tri_path_03_train.h5")
#print(model.summary())

with open('config.json') as config_file:
    config = json.load(config_file)
root = config['root']

patient_no = 210 
slice_no = 70 


with open('config.json') as config_file:
    config = json.load(config_file)
root = config['root']
patient_path = glob(root + '/*pat{}*'.format(patient_no))
patient_scans = utils.load_test_scans(patient_path[0])
patient_scans = utils.norm_test_scans(patient_scans)

test_slice = patient_scans[slice_no:slice_no+1,:,:,:4]
test_label = patient_scans[slice_no:slice_no+1,:,:,4]
scan = test_slice[0,:,:,1]
tmp_label = test_label[0]
print(test_slice.shape)


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
plt.savefig('outputs/test_prediction.png', bbox_inches='tight')
#plt.show()




y = test_label[0,15:223,15:223]

truth = y.reshape(43264,)
guess = pred.reshape(43264,)

print(classification_report(truth, guess, labels=[0,1,2,3,4]))











