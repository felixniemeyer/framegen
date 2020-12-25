from IPython import display

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time

from utils import * 

imgRes = Resolution(24, 16)
#imgRes = Resolution(48, 32)
batch_size = 16
        
print("loading files")

element_spec = tf.TensorSpec(shape=(None, imgRes.y, imgRes.x, 3), dtype=tf.float32, name='frame')

train_dataset = tf.data.experimental.load('data/0001-{}x{}.train.data'.format(imgRes.x, imgRes.y), element_spec, compression='GZIP')
test_dataset = tf.data.experimental.load('data/0001-{}x{}.test.data'.format(imgRes.x, imgRes.y), element_spec, compression='GZIP')

print("datasets prepared")

class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim, res):
        super(CVAE, self).__init__()
        
        (fcx, fcy, conv_layers) = self.calcLayerStuff(res.x, res.y)
        print("using {} conv layers".format(conv_layers))
        filters = 32
            
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tf.keras.layers.InputLayer(input_shape=(res.y, res.x, 3)))
        for i in range(conv_layers):
            self.encoder.add(tf.keras.layers.conv2d(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'))
            filters = filters * 2
        self.encoder.add(tf.keras.layers.Flatten())
        self.encoder.add(tf.keras.layers.Dense(latent_dim + latent_dim))

        self.decoder = tf.keras.Sequential()
        self.decoder.add(tf.keras.layers.InputLayer(input_shape=(latent_dim,)))
        self.decoder.add(tf.keras.layers.Dense(units=fcx*fcy*32, activation=tf.nn.relu))
        self.decoder.add(tf.keras.layers.Reshape(target_shape=(fcy, fcx, 32)))
        for i in range(conv_layers):
            self.decoder.add(tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'))
            filters = filters / 2
        # No activation
        tf.keras.layers.Conv2DTranspose(
            filters=3, kernel_size=3, strides=1, padding='same'),

    def calcLayerStuff(self, x,y):
        fcx = x
        fcy = y
        conv_layers = 0
        while(True):
            x = x / 2
            if x != int(x): 
                break
            y = y / 2
            if y != int(x): 
                break
            if x < 8 or y < 8:
                break
            fcx = x
            fcy = y
            conv_layers += 1
        return fcx, fcy, conv_layers
            

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x[:,:,:,:]), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

optimizer = tf.keras.optimizers.Adam(1e-4)


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

epochs = 300
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 8
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim, imgRes)


def generate_and_save_images(model, epoch, test_sample):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        # plt.imshow(predictions[i, :, :, :], cmap='gray')
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()
    
# Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]
generate_and_save_images(model, 0, test_sample)

for epoch in range(1, epochs + 1):
    print("\ntraining epoch", epoch, "\n\n")
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
    generate_and_save_images(model, epoch, test_sample)

def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))
plt.imshow(display_image(epoch))
plt.axis('off')    # Display images
