import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import signal
import sys

from utils import *  

from cvae import CVAE

# imgRes = Resolution(24, 16)
imgRes = Resolution(48, 32)
xxy = "{}x{}".format(imgRes.x, imgRes.y)

vid = cv2.VideoCapture('videos/0001-{}x{}.mp4'.format(imgRes.x, imgRes.y))

out = cv2.VideoWriter('results/videos/{}.mp4'.format(xxy), 
    cv2.VideoWriter_fourcc('m','p','4','v'),
    25, (imgRes.x, imgRes.y))
if(not out.isOpened()):
    print("failed to open video writer")
    exit(0)

latent_vars = 24
batch = 16
epochs = 200
model = CVAE(0,0,xcoders_load_prefick='models/{}-{}lat-{}batch-{}epoch/'.format(xxy, latent_vars, batch, epochs))

sequence_of_representations = []

end = False
def end_handler(sig, frame):
    global end 
    end = True
signal.signal(signal.SIGINT, end_handler)

previous_z = np.random.rand(1,24)
while(True):
    ret, inFrame = vid.read()
    if inFrame is None: 
        print("no frames left, terminating")
        #cv2.imshow('frame', frame)
        break
    
    frame_float = cv2.normalize(
        cv2.cvtColor(inFrame, cv2.COLOR_BGR2RGB), None, 
        alpha=0, beta=1, 
        norm_type=cv2.NORM_MINMAX, 
        dtype=cv2.CV_32F)
    
    a = np.array([frame_float])

    mean, logvar = model.encode(a) 
    z = tf.exp(logvar * .5) + mean
    
    sequence_of_representations.append(z)
    
    predictions = model.decode(z, True)[0].numpy()
    
    normalized_predictions = (predictions - np.min(predictions))/(np.max(predictions) - np.min(predictions)) # this set the range from 0 till 1
    outFrame = cv2.cvtColor((normalized_predictions * 255).astype(np.uint8), cv2.COLOR_BGR2RGB) # set is to a range from 0 till 255
    # cv2.imshow('', outFrame)
    out.write(outFrame)
    
    if end: 
        break
    
# np.save('repr_sequences/{}-{}lat-{}batch-{}epoch'.format(xxy, latent_vars, batch, epochs), sequence_of_representations)

out.release()

