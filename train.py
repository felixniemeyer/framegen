from IPython import display

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import time

from utils import * 
from cvae import CVAE

# imgRes = Resolution(24, 16)
imgRes = Resolution(48, 32)
latent_dim = 24
batch_size = 16
optimizer = tf.keras.optimizers.Adam(1e-4)
epochs = 200
num_examples_to_generate = 16

save_model = False 
        
element_spec = tf.TensorSpec(shape=(None, imgRes.y, imgRes.x, 3), dtype=tf.float32, name='frame')
train_dataset = tf.data.experimental.load('data/0001-{}x{}.train.data'.format(imgRes.x, imgRes.y), element_spec, compression='GZIP')
test_dataset = tf.data.experimental.load('data/0001-{}x{}.test.data'.format(imgRes.x, imgRes.y), element_spec, compression='GZIP')

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x[:,:,:,:])
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim, imgRes)

# Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(72): #random number for different pics 1337 is the las lol
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]
generate_and_save_images(model, 0, test_sample)

next = 1
for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        train_step(model, train_x, optimizer)
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset:
        loss(compute_loss(model, test_x))
    elbo = -loss.result()
    display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                .format(epoch, elbo, end_time - start_time))
    if(epoch == next or epoch == epochs): #generate only every
        generate_and_save_images(model, epoch, test_sample)
        next = next * 2

if save_model:
    model.save_xcoders("models/{}x{}-{}lat-{}batch-{}epoch/".format(imgRes.x, imgRes.y, latent_dim, batch_size, epochs))
