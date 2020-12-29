# takes as input some previous frames F_(t - {2^0, 2^1, 2^2, ... 2^h})
# predicts the frame F_t

import cv2
import tensorflow
import numpy as np
from utils import *

from cvae import CVAE 
    
#imgRes = Resolution(24, 16)
imgRes = Resolution(48, 32)
vid = cv2.VideoCapture('videos/0001-{}x{}.mp4'.format(imgRes.x, imgRes.y))
latent_dim = 24

model = CVAE(latent_dim, imgRes, 
    xcoders_prefick="models/{}x{}-{}lat-16batch-200epoch/".format(imgRes.x, imgRes.y, latent_dim))

train = 7
test = 3
train_images = []
test_images = []
batch_size = 16
i = 0
while(True):
    ret, frame = vid.read()
    if frame is None: 
        break
    
    cv2.imshow('frame', frame)

    frame_float = cv2.normalize(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None, 
        alpha=0, beta=1, 
        norm_type=cv2.NORM_MINMAX, 
        dtype=cv2.CV_32F)

    if (i % (train + test)) < train:
        train_images.append(frame_float)
    else: 
        test_images.append(frame_float)
    i += 1