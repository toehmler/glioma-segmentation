from keras import layers
from keras.models import model_from_json
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Lambda,Concatenate
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Add
from keras.layers import Maximum
from keras.models import Model
from keras import regularizers
from keras.optimizers import SGD
import keras.backend as K
from keras.regularizers import l1_l2
from losses import *

def tri_path(input_shape):
    X_input = Input(input_shape)

    local = Conv2D(64, (4,4),
            kernel_regularizer=l1_l2(l1 = 0.001, l2 = 0.001),
            strides=(1,1), padding='valid', activation='relu')(X_input)
    local = BatchNormalization()(local)
    local = Dropout(0.5)(local)


    local = Conv2D(64, (4,4),
            kernel_regularizer=l1_l2(l1 = 0.001, l2 = 0.001),
            strides=(1,1), padding='valid', activation='relu')(local)
    local = BatchNormalization()(local)
    local = Dropout(0.5)(local)

    local = Conv2D(64, (4,4),
            kernel_regularizer=l1_l2(l1 = 0.001, l2 = 0.001),
            strides=(1,1), padding='valid', activation='relu')(local)
    local = BatchNormalization()(local)
    local = Dropout(0.5)(local)

    local = Conv2D(64, (4,4),
            kernel_regularizer=l1_l2(l1 = 0.001, l2 = 0.001),
            strides=(1,1), padding='valid', activation='relu')(local)
    local = BatchNormalization()(local)
    local = Dropout(0.5)(local)

    local = Conv2D(64, (3,3),
            kernel_regularizer=l1_l2(l1 = 0.001, l2 = 0.001),
            strides=(1,1), padding='valid', activation='relu')(local)
    local = BatchNormalization()(local)
    local = Dropout(0.5)(local)

    local = Conv2D(64, (3,3),
            kernel_regularizer=l1_l2(l1 = 0.001, l2 = 0.001),
            strides=(1,1), padding='valid', activation='relu')(local)
    local = BatchNormalization()(local)
    local = Dropout(0.5)(local)




    inter = Conv2D(80, (7,7),
            kernel_regularizer=l1_l2(l1 = 0.001, l2 = 0.001),
            strides=(1,1), padding='valid', activation='relu')(X_input)
    inter = BatchNormalization()(inter)
    inter = Dropout(0.5)(inter)

    inter = Conv2D(80, (7,7),
            kernel_regularizer=l1_l2(l1 = 0.001, l2 = 0.001),
            strides=(1,1), padding='valid', activation='relu')(inter)
    inter = BatchNormalization()(inter)
    inter = Dropout(0.5)(inter)

    inter = Conv2D(80, (5,5),
            kernel_regularizer=l1_l2(l1 = 0.001, l2 = 0.001),
            strides=(1,1), padding='valid', activation='relu')(inter)
    inter = BatchNormalization()(inter)
    inter = Dropout(0.5)(inter)

    uni = Conv2D(160, (17,17),
            kernel_regularizer=l1_l2(l1 = 0.001, l2 = 0.001),
            strides=(1,1), padding='valid', activation='relu')(X_input)
    uni = BatchNormalization()(uni)
    uni = Dropout(0.25)(uni)

    out = Concatenate()([local, inter, uni])
    out = Conv2D(5,(21,21),strides=(1,1),padding='valid')(out)
    out = Activation('softmax')(out)

    model = Model(inputs=X_input, outputs=out)

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=[f1_score])

    return model

def two_path(input_shape):
    X_input = Input(input_shape)

    X = Conv2D(64,(7,7),strides=(1,1),padding='valid')(X_input)
    X = BatchNormalization()(X)
    X1 = Conv2D(64,(7,7),strides=(1,1),padding='valid')(X_input)
    X1 = BatchNormalization()(X1)
    X = layers.Maximum()([X,X1])
    X = Conv2D(64,(4,4),strides=(1,1),padding='valid',activation='relu')(X)

    X2 = Conv2D(160,(13,13),strides=(1,1),padding='valid')(X_input)
    X2 = BatchNormalization()(X2)
    X21 = Conv2D(160,(13,13),strides=(1,1),padding='valid')(X_input)
    X21 = BatchNormalization()(X21)
    X2 = layers.Maximum()([X2,X21])

    X3 = Conv2D(64,(3,3),strides=(1,1),padding='valid')(X)
    X3 = BatchNormalization()(X3)
    X31 =  Conv2D(64,(3,3),strides=(1,1),padding='valid')(X)
    X31 = BatchNormalization()(X31)
    X = layers.Maximum()([X3,X31])
    X = Conv2D(64,(2,2),strides=(1,1),padding='valid',activation='relu')(X)

    X = Concatenate()([X2,X])
    X = Conv2D(5,(21,21),strides=(1,1),padding='valid')(X)
    X = Activation('softmax')(X)

    # TODO switch to weighted categorical loss

    model = Model(inputs = X_input, outputs = X)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

if __name__ == "__main__":
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




    name = input('Model name: ')

#    with open("outputs/models/albert_test.json", "w") as json_file:
#        json_file.write(test_model_json)

    train_model = tri_path((37,37,4))
    train_model.save('outputs/models/{}_train.h5'.format(name))



    

    



    


