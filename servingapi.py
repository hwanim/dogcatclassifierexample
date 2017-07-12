import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
# from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
# from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
from urllib.request import urlopen
from microsoftbotframework import ReplyToActivity

IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic') # just so we remember which saved model is which, sizes must match
# TEST_DIR = ????




def make_response(message):
    # input_image = #input message
    # path = #extract the path of picture in input_image
    if message["attachments"][0]["contentType"] == "image/jpeg":
        ReplyToActivity(fill=message,text=catdogclassifiation(message)).send()


def catdogclassifiation(message):
    url = message["attachments"][0]["contentUrl"]
    data = url2img(url)
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model = model_load()
    model_out = model.predict([data])
    if np.argmax(model_out) == 1: return 'Dog'
    else: return 'Cat'

def url2img(url):
    resp = urlopen(url)
    img = np.asarray(bytearray(resp.read()), dtype ="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img)
    return img


def model_load():
    tf.reset_default_graph()


    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model = model.load(MODEL_NAME)

    return model


# def process_test_data():
#     testing_data = []
#     # for img in tqdm(os.listdir(TEST_DIR)):
#     #     path = os.path.join(TEST_DIR,img)
#     #     img_num = img.split('.')[0]
#     img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
#     testing_data.append([np.array(img), img_num])
#
#     # shuffle(testing_data)
#     # np.save('test_data.npy', testing_data)
#     return testing_data



# import matplotlib.pyplot as plt

# if you need to create the data:
# if you already have some saved:
#test_data = np.load('test_data.npy')

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
