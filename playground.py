
import numpy as np

from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Input, ReLU, UpSampling2D
#from keras.layers import BatchNormalization, Dropout,  ELU
from keras.layers import Add, Concatenate, Lambda, Dropout
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from PIL import Image
import glob
from matplotlib import pyplot as plt

import os



def upsample_neighbour(input_x):
    input_x_padded = K.spatial_2d_padding(input_x, padding=((2,2),(2,2)))
    width = K.int_shape(input_x)[1]
    height = K.int_shape(input_x)[2]
    output_x_list = []
    output_y_list = []
    for i_x in range(2, width + 2):
        for i_y in range(2, height + 2):
            output_y_list.append(input_x_padded[:,i_x-2:i_x+3,i_y-2:i_y+3,:])
        output_x_list.append(K.concatenate(output_y_list, axis=2))
        output_y_list = []
    return K.concatenate(output_x_list, axis=1)
    

def get_classification_model(input_shape=(128,64,3), print_summary=True, weight_decay=0.0005, dropout_rate=0):
    #input_shape=(160,60,3)
    x1_input = Input(shape=input_shape, name='input1')
    x2_input = Input(shape=input_shape, name='input2')
    
    
    conv1_tied = Conv2D(20, 5, kernel_regularizer=l2(weight_decay), activation="relu", name='conv1_tied')
    
    x1 = conv1_tied(x1_input)
    x2 = conv1_tied(x2_input)
    
    x1 = MaxPool2D(pool_size=2, padding='same', name='maxpool_f1')(x1)
    x2 = MaxPool2D(pool_size=2, padding='same', name='maxpool_g1')(x2)
    
    if dropout_rate > 0:
        x1 = Dropout(dropout_rate, name='dropout_x1_1')(x1)
        x2 = Dropout(dropout_rate, name='dropout_x2_1')(x2)
    
    conv2_tied = Conv2D(25, 5, kernel_regularizer=l2(weight_decay), activation="relu", name='conv2_tied')
    
    x1 = conv2_tied(x1)
    x2 = conv2_tied(x2)
    
    x1 = MaxPool2D(pool_size=2, padding='same', name='maxpool_f2')(x1)
    x2 = MaxPool2D(pool_size=2, padding='same', name='maxpool_g2')(x2)
    
    if dropout_rate > 0:
        x1 = Dropout(dropout_rate, name='dropout_x1_2')(x1)
        x2 = Dropout(dropout_rate, name='dropout_x2_2')(x2)
    
    
    f1 = UpSampling2D(size=(5,5), name='upsample_f1')(x1)
    g1 = Lambda(upsample_neighbour, name='upsample_neighbour_g1')(x2)
    g1 = Lambda(lambda x : -x, name='negate_g1')(g1)
    k1 = Add(name='add_k1')([f1, g1])
    
    f2 = UpSampling2D(size=(5,5), name='upsample_f2')(x2)
    g2 = Lambda(upsample_neighbour, name='upsample_neighbour_g2')(x1)
    g2 = Lambda(lambda x : -x, name='negate_g2')(g2)
    k2 = Add(name='add_k2')([f2, g2])
    
    k1 = ReLU(name='relu_k1')(k1)
    k2 = ReLU(name='relu_k2')(k2)
    
    k1 = Conv2D(25, 5, strides=(5,5), kernel_regularizer=l2(weight_decay), activation="relu", name='patch_summary_k1')(k1)
    k2 = Conv2D(25, 5, strides=(5,5), kernel_regularizer=l2(weight_decay), activation="relu", name='patch_summary_k2')(k2)
    
    k1 = Conv2D(25, 3, kernel_regularizer=l2(weight_decay), activation="relu", name='accross_patch_conv_k1')(k1)
    k2 = Conv2D(25, 3, kernel_regularizer=l2(weight_decay), activation="relu", name='accross_patch_conv_k2')(k2)
    
    k1 = MaxPool2D(pool_size=2, padding='same', name='maxpool_k1')(k1)
    k2 = MaxPool2D(pool_size=2, padding='same', name='maxpool_k2')(k2)
    
    if dropout_rate > 0:
        k1 = Dropout(dropout_rate, name='dropout_k1')(k1)
        k2 = Dropout(dropout_rate, name='dropout_k2')(k2)
    
    k1 = Flatten(name='flatten_k1')(k1)
    k2 = Flatten(name='flatten_k2')(k2)
    
    
    y = Concatenate(name='concat_k1_and_k2')([k1, k2])
    
    y = Dense(units=500, kernel_regularizer=l2(weight_decay), activation="relu")(y)
    
    if dropout_rate > 0:
        y = Dropout(dropout_rate, name='dropout_y')(y)

    y_out = Dense(units=2, kernel_regularizer=l2(weight_decay), activation="softmax")(y)
    
    model = Model(inputs=[x1_input, x2_input], outputs=[y_out])

    if print_summary == True:
        model.summary()
        
    return model


def prepare_dataset(file_path = '../../Fax/MasinskoUcenje/ml_all/images/detected/', img_format='jpg'):
    #prepare_dataset
    filePaths = file_path + '*.' + img_format
    
    distinct_identities = 1467
    all_images = [list() for _ in range(distinct_identities)]

    for filename in sorted(glob.glob(filePaths)):
        identity_num = int(filename.split('/')[-1].split('_')[0])

        assert identity_num < distinct_identities

        im=Image.open(filename)
        all_images[identity_num].append(np.array(im)/255.0)

    return all_images


def generate_random_train(all_images, batch_size_big=64):
    batch_size = batch_size_big//2
    while True:
        np.random.rand(0,len(all_images), batch_size//2)
        left_imgs = np.empty([2*batch_size, 128, 64, 3])
        right_imgs = np.empty([2*batch_size, 128, 64, 3])
        y = np.empty([2*batch_size, 2])
        
        
        a = np.arange(1000)
        np.random.shuffle(a)
        inds = a[:batch_size]        
        for batch_ind, cur_img_ind in enumerate(inds):
            b = np.arange(len(all_images[cur_img_ind]))
            np.random.shuffle(b)
            inds_same = b[:2]
            #print(batch_ind, cur_img_ind, inds_same[0])
            left_imgs[batch_ind]
            all_images[cur_img_ind]
            all_images[cur_img_ind][inds_same[0]]
            left_imgs[batch_ind] = all_images[cur_img_ind][inds_same[0]]
            right_imgs[batch_ind] = all_images[cur_img_ind][inds_same[1]]
            y[batch_ind] = np.array([1,0])
            
        a = np.arange(1000)
        np.random.shuffle(a)
        inds = a[:2*batch_size]        
        for batch_ind in range(0,2*batch_size, 2):
            left_imgs[batch_size + batch_ind//2] = all_images[inds[batch_ind]][np.random.randint(len(all_images[inds[batch_ind]]))]
            right_imgs[batch_size + batch_ind//2] = all_images[inds[batch_ind+1]][np.random.randint(len(all_images[inds[batch_ind+1]]))]
            y[batch_size + batch_ind//2] = np.array([0,1])
            
        rng_state = np.random.get_state()
        np.random.shuffle(left_imgs)
        np.random.set_state(rng_state)
        np.random.shuffle(right_imgs)
        np.random.set_state(rng_state)
        np.random.shuffle(y)
        
        yield [left_imgs, right_imgs], y


def generate_random_val(all_images, batch_size_big=64):
    batch_size = batch_size_big//2
    while True:
        np.random.rand(0,len(all_images), batch_size//2)
        left_imgs = np.empty([batch_size*2, 128, 64, 3])
        right_imgs = np.empty([batch_size*2, 128, 64, 3])
        y = np.empty([2*batch_size, 2])
        
        
        a = np.arange(1000, 1200, 1)
        np.random.shuffle(a)
        inds = a[:batch_size]        
        for batch_ind, cur_img_ind in enumerate(inds):
            b = np.arange(len(all_images[cur_img_ind]))
            np.random.shuffle(b)
            inds_same = b[:2]
            #print(batch_ind, cur_img_ind, inds_same[0])
            left_imgs[batch_ind]
            all_images[cur_img_ind]
            all_images[cur_img_ind][inds_same[0]]
            left_imgs[batch_ind] = all_images[cur_img_ind][inds_same[0]]
            right_imgs[batch_ind] = all_images[cur_img_ind][inds_same[1]]
            y[batch_ind] = np.array([1,0])
            
        a = np.arange(1000, 1200, 1)
        np.random.shuffle(a)
        inds = a[:2*batch_size]        
        for batch_ind in range(0,2*batch_size, 2):
            left_imgs[batch_size + batch_ind//2] = all_images[inds[batch_ind]][np.random.randint(len(all_images[inds[batch_ind]]))]
            right_imgs[batch_size + batch_ind//2] = all_images[inds[batch_ind+1]][np.random.randint(len(all_images[inds[batch_ind+1]]))]
            y[batch_size + batch_ind//2] = np.array([0,1])
            
        rng_state = np.random.get_state()
        np.random.shuffle(left_imgs)
        np.random.set_state(rng_state)
        np.random.shuffle(right_imgs)
        np.random.set_state(rng_state)
        np.random.shuffle(y)
        
        yield [left_imgs, right_imgs], y



if __name__ == '__main__':
    # TODO if model_path argument then load model from weights else train new model
    # parser = argparse.ArgumentParser()
    # help_ = "Load h5 model trained weights"
    # parser.add_argument("-w", "--weights", help=help_)
    # args = parser.parse_args()

    all_images = prepare_dataset()
    model = get_classification_model(dropout_rate=0.5)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    gen_tr = generate_random_train(all_images=all_images, batch_size_big=60)
    gen_val = generate_random_val(all_images=all_images, batch_size_big=60)

    model_checkpoint_dir = 'models_dp05_epoch200'
    model_checkpoint_name = 'dropout_05'

    checkpoint_path = os.path.join(model_checkpoint_dir, model_checkpoint_name + '.{epoch:02d}-{val_loss:.2f}.hdf5')

    epochs = 150
    history = model.fit_generator(generator=gen_tr,
                                  steps_per_epoch=200,
                                  epochs=epochs,
                                  validation_data=gen_val,
                                  validation_steps=1,
                                  callbacks = [ModelCheckpoint(save_best_only=False,
                                                               filepath=checkpoint_path),
                                               # ReduceLROnPlateau(factor=0.2, verbose=1),
                                               TensorBoard(log_dir='logs')])


    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(range(epochs), loss, color='red', label='training')
    plt.plot(range(epochs), val_loss, color='orange', label='validation')
    plt.legend(loc='best')
    plt.show()


    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.plot(range(epochs), acc, color='red', label='training')
    plt.plot(range(epochs), val_acc, color='orange', label='validation')
    plt.legend(loc='best')
    plt.show()

