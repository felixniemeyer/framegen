import cv2
import tensorflow
import numpy as np
from utils import *
    
#imgRes = Resolution(24, 16)
imgRes = Resolution(48, 32)
vid = cv2.VideoCapture('videos/0001-{}x{}.mp4'.format(imgRes.x, imgRes.y))
train = 7
test = 3
train_images = []
test_images = []
batch_size = 16
i = 0
while(True):
    ret, frame = vid.read()
    if frame is None: 
        #cv2.imshow('frame', frame)
        break
    
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_float = cv2.normalize(frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    if (i % (train + test)) < train:
        train_images.append(frame_float)
    else: 
        test_images.append(frame_float)
    i += 1
        
# np.save('0001-{}x{}.train.data'.format(imgRes.x, imgRes.y), train_dataset)
# np.save('0001-{}x{}.test.data'.format(imgRes.x, imgRes.y), test_dataset)

buffer_size = 40000

train_dataset = tensorflow.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)
tensorflow.data.experimental.save(train_dataset, 'data/0001-{}x{}.train.data'.format(imgRes.x, imgRes.y), compression='GZIP')
print("train element spec", train_dataset.element_spec)
test_dataset = tensorflow.data.Dataset.from_tensor_slices(test_images).shuffle(buffer_size).batch(batch_size)
tensorflow.data.experimental.save(test_dataset, 'data/0001-{}x{}.test.data'.format(imgRes.x, imgRes.y), compression='GZIP')
print("test element spec", test_dataset.element_spec)

