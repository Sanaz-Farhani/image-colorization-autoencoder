from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.utils import plot_model
from tensorflow.keras import backend as K
from keras.utils import img_to_array
from keras.utils import array_to_img 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os

image_dir = 'saved_image'
save_dir = os.path.join(os.getcwd(), image_dir)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#input image dimensions 
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
channels = x_train.shape[3]

imgs = x_test[:100]
imgs = imgs.reshape((10,10, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Test color img') 
plt.imshow(imgs, interpolation='none')
plt.savefig('%s/test_color.png' % image_dir)
# plt.show()
 
x_train_gray = rgb2gray(x_train)
x_test_gray = rgb2gray(x_test)

#display grayscale version of test image 
image = x_test_gray[:100]
image = image.reshape((10,10, img_rows, img_cols))
image = np.vstack([np.hstack(i) for i in image])
plt.figure()
plt.axis('off')
plt.title('Test gray img (Input)') 
plt.imshow(image, interpolation='none', cmap='gray')
plt.savefig('%s/test_gray.png' % image_dir)

# plt.show()

#normalized input train and test color images 
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#normalized input train and test grayscale images 
x_test_gray = x_test_gray.astype('float32')/255
x_train_gray = x_train_gray.astype('float32')/255

#reshape images to rows x col x channel for  CNN output/validation
x_train = x_train.reshape(x_train.shape[0], img_rows,img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows,img_cols, channels)

#reshape images to row x clo x channel for CNN input 
x_train_gray = x_train_gray.reshape(x_train_gray.shape[0], img_rows, img_cols,1)
x_test_gray = x_test_gray.reshape(x_test_gray.shape[0], img_rows, img_cols,1)

#network parameter 
input_shape = (img_rows, img_cols, 1)
batch_size = 32
kernel_size = 3
latent_dim = 256

#encoder/decoder number of CNN layers and filters per player
layer_filters = [64, 128, 256]

#build the autoencoder model 
#first build the encoder model
inputs = Input(shape= input_shape, name= 'encoder_input')
x= inputs
#stack of Conv2D(64)-Conv2D(128)-Conv2D(256)
for filters in layer_filters:
    x = Conv2D(filters= filters,
               kernel_size= kernel_size,
               strides=2,
               activation='relu',
               padding= 'same')(x) 
    
#shape info needed to build decoder model so we don't do hand computation
#the input to the decoder's first Conv2DTranspose will have this shape
#shape is (4,4,256) which is processed by the decoder back to (32,32,3)
shape = K.int_shape(x)

#generate a latent  vector 
x = Flatten()(x)
latent = Dense(latent_dim, name= 'latent_vector')(x)

# instantiate encoder model
encoder = Model(inputs, latent, name= 'encoder')
# encoder.summary()

#build decoder model 
latent_inputs = Input(shape=(latent_dim,), name= 'decoder_input')
x= Dense(shape[1]* shape[2]* shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)
#stack of Conv2DTranspose(256)-Conv2Dtranspose(128)-Conv2DTranspose(64)
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters= filters,
                        kernel_size = kernel_size,
                        strides=2, 
                        activation='relu',
                        padding='same')(x)

output = Conv2DTranspose(filters= channels,
                        kernel_size = kernel_size,
                        activation= 'sigmoid',
                        padding= 'same',
                        name= 'decoder_output')(x)

#instantiate decoder model
decoder = Model(latent_inputs, output, name='decoder')
decoder.summary()

# autoencoder = encoder + decoder 
# instantiate = autoencoder model
autoencoder = Model(inputs, decoder(encoder(inputs)), name= 'autoencoder')
autoencoder.summary()

#prepare model saving directory
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'colorized_as_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

filepath = os.path.join(save_dir, model_name)

#reduce learning rate by sqrt(0.1) if the loss does not improve in 5 epochs
lr_reducer = ReduceLROnPlateau(factor= np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               verbose=1,
                               min_lr=0.5e-6)

#save weights for future use(e.g. reload parameters w/o training)
check_point = ModelCheckpoint(filepath= filepath,
                              monitor= 'val_loss',
                              verbose=1,
                              save_best_only=True)

autoencoder.compile(loss= 'mse', optimizer= 'adam')

#called every epochs 
callbacks = [lr_reducer, check_point]

#train the autoencoder
autoencoder.fit(x_train_gray, x_train, validation_data=(x_test_gray, x_test),
                epochs=30,
                batch_size= batch_size,
                callbacks= callbacks)

#predict the autoencoder  output from test data
x_decoded = autoencoder.predict(x_test_gray)
plt.imshow(array_to_img(x_test_gray[0])) 
plt.imshow(array_to_img(x_decoded[0]))

for i in range(0,3):
    fig = plt.figure()
    ax1 = fig.add_subplot(3,3,1)
    ax1.set_title('Gray Scale')
    ax1.imshow(array_to_img(x_test_gray[i]))

    ax2 = fig.add_subplot(3,3,2)
    ax2.set_title('Ground_Truth')
    ax2.imshow(array_to_img(x_test[i]))

    ax3 = fig.add_subplot(3,3,3)
    ax3.set_title('Converted RGB')
    ax3.imshow(array_to_img(x_decoded[i]))
    
plt.show()    

#display the 1st 100 colorized images 
imgs = x_decoded[:100]
imgs = imgs.reshape((10,10, img_rows, img_cols, channels))
imgs = np.vstack([np.hstack(i) for i in imgs])
plt.figure()
plt.axis('off')
plt.title('Colorized test images (Preicted)')
plt.imshow(imgs, interpolation='none')
plt.savefig('%s/colorized.png'%image_dir)
plt.show()