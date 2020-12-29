import tensorflow as tf
from tensorflow.keras import layers 

class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim, res, filters=32, xcoders_load_prefick=None):
        super(CVAE, self).__init__()
        
        if xcoders_load_prefick != None: 
            self.load_xcoders(xcoders_load_prefick)
        else:
            (fcx, fcy, conv_layers) = self.calcLayerStuff(res.x, res.y)
            print("using {} conv layers".format(conv_layers))
            print("layer 0 size: {}x{}".format(fcx, fcy))
            print("filters: {}*2**(layer_id + 1)".format(filters))
                
            self.latent_dim = latent_dim
            self.encoder = tf.keras.Sequential()
            self.encoder.add(layers.InputLayer(input_shape=(res.y, res.x, 3)))
            for i in range(conv_layers):
                self.encoder.add(layers.Conv2D(
                    filters=filters, kernel_size=3, strides=(2, 2), activation='relu'))
                filters = filters * 2
            self.encoder.add(layers.Flatten())
            self.encoder.add(layers.Dense(latent_dim + latent_dim))

            self.decoder = tf.keras.Sequential()
            self.decoder.add(layers.InputLayer(input_shape=(latent_dim,)))
            self.decoder.add(layers.Dense(units=fcx*fcy*32, activation=tf.nn.relu))
            self.decoder.add(layers.Reshape(target_shape=(fcy, fcx, 32)))
            for i in range(conv_layers):
                self.decoder.add(layers.Conv2DTranspose(
                    filters=filters, kernel_size=3, strides=2, padding='same',
                    activation='relu'))
                filters = int(filters / 2)
            # No activation
            self.decoder.add(layers.Conv2DTranspose(
                filters=3, kernel_size=3, strides=1, padding='same'))

    def calcLayerStuff(self, x,y):
        fcx = x
        fcy = y
        conv_layers = 0
        while(True):
            x = x / 2
            if x != int(x): 
                break
            y = y / 2
            if y != int(y): 
                break
            if x < 4 or y < 4:
                break
            fcx = int(x)
            fcy = int(y)
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

    def save_xcoders(self, prefick): 
        self.encoder.save(prefick + "encoder")
        self.decoder.save(prefick + "decoder")
        
    def load_xcoders(self, prefick):
        self.encoder = tf.keras.models.load_model(prefick + "encoder")
        self.decoder = tf.keras.models.load_model(prefick + "decoder")