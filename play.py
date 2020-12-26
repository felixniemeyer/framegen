import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import signal
import sys

from utils import *  

# imgRes = Resolution(24, 16)
imgRes = Resolution(48, 32)
xxy = "{}x{}".format(imgRes.x, imgRes.y)

out = cv2.VideoWriter('{}'.format(xxy), 
    cv2.VideoWriter_fourcc('m','p','4','v'),
    24, (1024, 768))
if(not out.isOpened()):
    print("failed to open video writer")
    exit(0)

decoder = tf.keras.models.load_model('models/{}-24lat-16batch-200epoch/decoder'.format(xxy))
encoder = tf.keras.models.load_model('models/{}-24lat-16batch-200epoch/encoder'.format(xxy))

element_spec = tf.TensorSpec(shape=(None, imgRes.y, imgRes.x, 3), dtype=tf.float32, name='frame')
test_dataset = tf.data.experimental.load('data/0001-{}.test.data'.format(xxy), element_spec, compression='GZIP')

end = False
def end_handler(sig, frame):
    global end 
    end = True
signal.signal(signal.SIGINT, end_handler)

previous_z = np.random.rand(1,24)
for img in test_dataset.take(int(random.random()*1336)):
    mean, logvar = tf.split(encoder(img[0:1,:,:,:]), num_or_size_splits=2, axis=1)
    eps = tf.random.normal(shape=mean.shape)
    z = eps * tf.exp(logvar * .5) + mean
    for i in range(32): 
        p = i / 32
        weighted_z =  p * z + (1-p) * previous_z
        predictions = tf.sigmoid(decoder(weighted_z))[0].numpy()
        normalized_predictions = (predictions - np.min(predictions))/(np.max(predictions) - np.min(predictions)) # this set the range from 0 till 1
        frame = (normalized_predictions * 255).astype(np.uint8) # set is to a range from 0 till 255
        print(frame.shape())
        out.write(frame)
       # plt.imshow(predictions[0])
       # plt.show()
       # plt.clf()
        if end: 
            break
    previous_z = z
    if end: 
        break

out.release()

