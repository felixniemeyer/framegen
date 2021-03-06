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

out = cv2.VideoWriter('result-video-{}.mp4'.format(xxy), 
    cv2.VideoWriter_fourcc('m','p','4','v'),
    24, (imgRes.x, imgRes.y))
if(not out.isOpened()):
    print("failed to open video writer")
    exit(0)

decoder = tf.keras.models.load_model('models/{}-24lat-16batch-10epoch/decoder'.format(xxy), compile=False)
encoder = tf.keras.models.load_model('models/{}-24lat-16batch-10epoch/encoder'.format(xxy), compile=False)

element_spec = tf.TensorSpec(shape=(None, imgRes.y, imgRes.x, 3), dtype=tf.float32, name='frame')
test_dataset = tf.data.experimental.load('data/{}.test.data'.format(xxy), element_spec, compression='GZIP')

end = False
def end_handler(sig, frame):
    global end 
    end = True
signal.signal(signal.SIGINT, end_handler)

previous_z = np.random.rand(1,24)
interpolations = 32
f = 1000
for img in test_dataset.take(6):
    f = f+1
    mean, logvar = tf.split(encoder(img[0:1,:,:,:]), num_or_size_splits=2, axis=1)
    eps = tf.random.normal(shape=mean.shape)
    z = eps * tf.exp(logvar * .5) + mean
    for i in range(interpolations): 
        p = i / interpolations
        weighted_z =  p * z + (1-p) * previous_z
        predictions = tf.sigmoid(decoder(weighted_z))[0].numpy()
        normalized_predictions = (predictions - np.min(predictions))/(np.max(predictions) - np.min(predictions)) # this set the range from 0 till 1
        frame = (normalized_predictions * 255).astype(np.uint8) # set is to a range from 0 till 255
        out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
       # plt.imshow(predictions[0])
       # plt.show()
       # plt.clf()
        if end: 
            break
    previous_z = z
    
    if f % 10 == 0:
        print(int((f-1000) * 32 / 24), "s video generated")
    
    if end: 
        break

out.release()

