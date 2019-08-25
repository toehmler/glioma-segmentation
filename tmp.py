from keras.models import load_model
from keras.models import model_from_json

# load json and create model
json_file = open('outputs/models/tri_path_test.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
print("Loaded model from disk")
print(loaded_model.summary())
