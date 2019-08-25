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
model.load_weights("outputs/models/tri_path_02_train.h5")
print(model.summary())

with open('config.json') as config_file:
    config = json.load(config_file)
root = config['root']

patient_no = 190
slice_no = 80


with open('config.json') as config_file:
    config = json.load(config_file)
root = config['root']
patient_path = glob(root + '/*pat{}*'.format(patient_no))
patient_scans = utils.load_test_scans(patient_path[0])
patient_scans = utils.norm_test_scans(patient_scans)

test_slice = patient_scans[slice_no:slice_no,:,:,:4]
test_label = patient_scans[slice_no:slice_no,:,:,4]


prediction = model.predict_classes(test_slice)
print(prediction.shape)

y = test_label[15:223,15:223]
truth = y.reshape(43264,)
print(classification_report(truth, prediction, labels=[0,1,2,3,4]))










