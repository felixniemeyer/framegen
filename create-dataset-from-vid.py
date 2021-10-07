import cv2
import tensorflow
import numpy as np
from utils import *
    
imgRes = Resolution(48, 32)
vid = cv2.VideoCapture('scaled-video.mp4')
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
        
# np.save('0001-{}x{}.train.data'.format(imgRes.x, imgRes.y), train_dataset)
# np.save('0001-{}x{}.test.data'.format(imgRes.x, imgRes.y), test_dataset)

buffer_size = 4000

train_dataset = tensorflow.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)
tensorflow.data.experimental.save(train_dataset, 'data/{}x{}.train.data'.format(imgRes.x, imgRes.y), compression='GZIP')
print("train element spec", train_dataset.element_spec)
test_dataset = tensorflow.data.Dataset.from_tensor_slices(test_images).shuffle(buffer_size).batch(batch_size)
tensorflow.data.experimental.save(test_dataset, 'data/{}x{}.test.data'.format(imgRes.x, imgRes.y), compression='GZIP')
print("test element spec", test_dataset.element_spec)

