import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic') # just so we remember which saved model is which, sizes must match
# TEST_DIR = ????


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')


def process_test_data():
    testing_data = []
    # for img in tqdm(os.listdir(TEST_DIR)):
    #     path = os.path.join(TEST_DIR,img)
    #     img_num = img.split('.')[0]
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    testing_data.append([np.array(img), img_num])

    # shuffle(testing_data)
    # np.save('test_data.npy', testing_data)
    return testing_data



import matplotlib.pyplot as plt

# if you need to create the data:
# if you already have some saved:
#test_data = np.load('test_data.npy')
def catdogclassifiation():
    test_data = process_test_data()
    img_data = test_data[0]
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[0]
        if np.argmax(model_out) == 1: return 'Dog'
        else: return 'Cat'

# fig=plt.figure()


# for num,data in enumerate(test_data[:12]):
#     # cat: [1,0]
#     # dog: [0,1]
#
#     img_num = data[1]
#     img_data = data[0]
#
#     y = fig.add_subplot(3,4,num+1)
#     orig = img_data
#     data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
#     #model_out = model.predict([data])[0]
#     model_out = model.predict([data])[0]
#
#     if np.argmax(model_out) == 1: str_label='Dog'
#     else: str_label='Cat'
#
#     y.imshow(orig,cmap='gray')
#     plt.title(str_label)
#     y.axes.get_xaxis().set_visible(False)
#     y.axes.get_yaxis().set_visible(False)
# plt.show()
#
#
# with open('submission_file.csv','w') as f:
#     f.write('id,label\n')
#
# with open('submission_file.csv','a') as f:
#     for data in tqdm(test_data):
#         img_num = data[1]
#         img_data = data[0]
#         orig = img_data
#         data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
#         model_out = model.predict([data])[0]
#         f.write('{},{}\n'.format(img_num,model_out[1]))
