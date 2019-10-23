from keras.models import load_model
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
model.load_weights("outputs/models/tri_path_01_train.h5")
#print(model.summary())

with open('config.json') as config_file:
    config = json.load(config_file)
root = config['root']

patient_no = 205
slice_no = 70


with open('config.json') as config_file:
    config = json.load(config_file)
root = config['root']
patient_path = glob(root + '/*pat{}*'.format(patient_no))
patient_scans = utils.load_test_scans(patient_path[0])
patient_scans = utils.norm_test_scans(patient_scans)

test_slice = patient_scans[slice_no:slice_no+1,:,:,:4]
test_label = patient_scans[slice_no:slice_no+1,:,:,4]

prediction = model.predict(test_slice, verbose=1)
prediction = np.argmax(prediction, axis=-1)
pred = prediction[0]


y = test_label[0,15:223,15:223]

truth = y.reshape(43264,)
guess = pred.reshape(43264,)

print(classification_report(truth, guess, labels=[0,1,2,3,4]))












