import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

import models

model_data_path = 'NYU_FCRN.ckpt'
fileAirPath = './image/'
airPic = os.listdir(fileAirPath)
airPic.sort()

fileRoadPath = './pic'
roadPic = os.listdir(fileRoadPath)
roadPic = [int(fi.split('.')[0][14:]) for fi in roadPic]
roadPic.sort()



# Default input size
height = 228
width = 304
channels = 3
batch_size = 1

# Read image
imgAll = []
for file in airPic:
    img = Image.open(fileAirPath + fileAirPath
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)
    imgAll.append(img)

# Create a placeholder for the input image
input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

# Construct the network
net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

with tf.Session() as sess:

    # Load the converted parameters
    print('Loading the model')

    # Use to load from ckpt file
    saver = tf.train.Saver()     
    saver.restore(sess, model_data_path)

    # Use to load from npy file
    #net.load(model_data_path, sess) 

    # Evalute the network for the given image
    
    for img in imgAll:
        pred = sess.run(net.get_output(), feed_dict={input_node: img})

        predImg = pred[0,:,:,0]       
        print(predImg.shape)
        break
    
    # Plot result
#     fig = plt.figure()
#     ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
#     fig.colorbar(ii)
#     plt.show()


