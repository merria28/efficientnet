import tensorflow as tf
# if you use tensorflow.keras: 
from efficientnet.tfkeras import EfficientNetB0
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras import backend as K
import cv2
import numpy as np
import math

import os
import os.path as osp
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0"

training_data_path = '../GenerateRecapDataset/train/train.txt'
test_data_path = '../GenerateRecapDataset/test/test.txt'

def generate_arrays_from_file(path ,input_size=64, batch_size=128):
    data=np.loadtxt(path,dtype='str',delimiter=' ')
  
    while True:
        np.random.shuffle(data)
        images = []
        image_fns = []
        for i in data:
            try:
                im_fn = i[0]
                label = i[1]
                label = int(label)
                ohl=np_utils.to_categorical(label,num_classes=2)
                im_path = 'F:\\vehicle_recapture\\GenerateRecapDataset\\'+im_fn            
                im = cv2.imread(im_path)
                im = im.astype(np.float32)

                #規一化
                im /= 255
                im -= [0.4465,0.4822,0.4914]
                im /= [0.2010,0.1994,0.2023]

                images.append(im)
                image_fns.append(ohl)
                
                if len(images) == batch_size:
                    yield np.array(images), np.array(image_fns)
                    images = []
                    image_fns = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue

n_classes = 2
batch_size = 128
nb_epoch = 25
data_augmentation = True
tdata = np.loadtxt(training_data_path,dtype='str',delimiter=' ')
vdata = np.loadtxt(test_data_path,dtype='str',delimiter=' ')
train_num = tdata.size >> 1
validation_num = vdata.size >> 1

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('EfficientNetB4.csv')

# build model
#base_model = EfficientNetB0(include_top=False, weights='imagenet')

# 注意 : efficientnet/model.py 第 238 行 has_se = False
# 因為 opencv dnn 調用不支持 Squeeze and Excitation phase
# 將來 h5 轉 pb freeze model 時也要設 has_se = False

base_model = EfficientNetB0(include_top=False, weights=None)
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])
model.summary()

# train
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_data = generate_arrays_from_file(path = training_data_path,batch_size=batch_size)
steps_per_epoch=math.ceil(train_num/batch_size) 
validation_data=generate_arrays_from_file(path = test_data_path,batch_size=batch_size)
validation_steps= math.ceil(validation_num/batch_size)
model.fit_generator(generator = train_data,steps_per_epoch=steps_per_epoch, epochs=nb_epoch,callbacks=[lr_reducer, early_stopper, csv_logger],
    validation_data = validation_data,validation_steps = validation_steps,max_queue_size=10,workers=1)

model.save('model_efficientB0.h5')