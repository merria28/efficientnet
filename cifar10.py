"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping

import numpy as np
import resnet
import math
import cv2
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet50_cifar2.csv')

batch_size = 128
nb_classes = 2
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 64, 64
# The CIFAR10 images are RGB.
img_channels = 3

# The data, shuffled and split between train and test sets:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)

#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')

# subtract mean and normalize
#mean_image = np.mean(X_train, axis=0)
#X_train -= mean_image
#X_test -= mean_image
#X_train /= 128.
#X_test /= 128.
train_num = 610000
validation_num = 207000
model = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


training_data_path = '../GenerateRecapDataset/train/train.txt'
test_data_path = '../GenerateRecapDataset/test/test.txt'
def generate_arrays_from_file(path ,input_size=64, batch_size=128):
    data=np.loadtxt(path,dtype='str',delimiter=' ')
    print('the total number of traindata:',data.size)
    np.random.shuffle(data)
    print('{} training images in {}'.format(
        data.size, path))
    while True:
        np.random.shuffle(data)
        images = []
        image_fns = []
        cn = 0
        for i in data:
            try:
                im_fn = i[0]
                label = i[1]
                #print(label)
                label=int(label)
                ohl=np_utils.to_categorical(label,num_classes=2)
                im_path = 'H:\\vehicle_recapture\\GenerateRecapDataset\\'+im_fn            
                im = cv2.imread(im_path)
                im = im.astype(np.float32)
                im /= 255
                #im = image.img_to_array(im)
                # im = np.expand_dims(im, axis=0)
                # print(im.shape)
                images.append(im)
                image_fns.append(ohl)
                
                if len(images) == batch_size:
                    #print("the image numbers:",len(images))
                    #print("the flag numbers:",len(image_fns))
                   # print(image_fns)
                    yield np.array(images), np.array(image_fns)
                    images = []
                    image_fns = []
                    cn=0
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              callbacks=[lr_reducer, early_stopper, csv_logger])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
  #  datagen = ImageDataGenerator(
   #     featurewise_center=False,  # set input mean to 0 over the dataset
    #    samplewise_center=False,  # set each sample mean to 0
     #   featurewise_std_normalization=False,  # divide inputs by std of the dataset
      #  samplewise_std_normalization=False,  # divide each input by its std
       # zca_whitening=False,  # apply ZCA whitening
       # rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
   #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
   #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
   #     horizontal_flip=True,  # randomly flip images
  #      vertical_flip=False)  # randomly flip images

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    #datagen.fit(X_train)

    train_data = generate_arrays_from_file(path = training_data_path,batch_size=batch_size)
    steps_per_epoch=math.ceil(train_num/batch_size) 
    epochs = 5344
    validation_data=generate_arrays_from_file(path = test_data_path,batch_size=batch_size)
    validation_steps= math.ceil(validation_num/batch_size)
    model.fit_generator(generator = train_data,steps_per_epoch=steps_per_epoch, epochs=epochs,callbacks=[lr_reducer, early_stopper, csv_logger],
            validation_data = validation_data,validation_steps = validation_steps,max_queue_size=10,workers=1)
    # Fit the model on the batches generated by datagen.flow().
   # model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
      #                  steps_per_epoch=X_train.shape[0] // batch_size,
      #                  validation_data=(X_test, Y_test),
      #                  epochs=nb_epoch, verbose=1, max_q_size=100,
       #                 callbacks=[lr_reducer, early_stopper, csv_logger])
    model.save('model.h5')