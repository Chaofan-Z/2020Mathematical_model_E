import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

import models

data = pd.read_csv('./amos0313_last.csv')

def getlabel(date):
    #     比如030200，代表3点2分
    hour = int(date[:2])
    minute = int(date[2:4])

    idNum = hour * 60 + minute + 959
    #     print(idNum)
    try:
        label = data.at[idNum, 'MOR']
    #         print(label)
    except:
        print(hour, minute, idNum)

    return label



def predict():
    model_data_path = 'NYU_FCRN.ckpt'
    fileAirPath = './image/'
    airPic = os.listdir(fileAirPath)
    airPic.sort()

    fileRoadPath = './pic/'
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
    imgSumAll = []

    for file in airPic:
        img = Image.open(fileAirPath + file)

    # for file in roadPic:
    #     img = Image.open(fileRoadPath + "original_frame" + str(file) + '.bmp')

        img = img.resize([width, height], Image.ANTIALIAS)
        img = np.array(img).astype('float32')
        img = np.expand_dims(np.asarray(img), axis=0)
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
            predImg = pred[0, :, :, 0]
            imgSum = predImg.sum()
            imgSumAll.append((imgSum/100))
            # break


        temp = []
        for i in range(24):
            for j in range(60):
                if j == 0:
                    date = str(i) + ":0" + str(j)
                else:
                    #             date = str(i) + ":" + str(j)
                    date = ""
                temp.append(date)
        datetime = temp[:480]
        len(datetime)

        plt.figure(figsize=(15, 7), dpi=150)
        plt.xticks(np.arange(len(datetime)), datetime, rotation=45)
        plt.plot(imgSumAll, label="Image depth")

        plt.xlabel("Time", fontsize=25)
        plt.ylabel("Image depth", fontsize=25)

        plt.tick_params(labelsize=15)
        plt.legend(fontsize=20)
        plt.grid(which='major', axis='y')

        plt.savefig('./airDepth.jpg', bbox_inches='tight')

        plt.show()
        # imgSumAll = '\t'.join()

        # # Plot result
        # fig = plt.figure()
        # ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        # fig.colorbar(ii)
        # plt.show()

        # return pred
        
                
def main():

    pred = predict()
    
    os._exit(0)

if __name__ == '__main__':
    main()

        



