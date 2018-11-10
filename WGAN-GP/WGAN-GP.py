import numpy as np
import matplotlib.pyplot as plt
import os
from functools import partial

from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, ZeroPadding2D, Dropout, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from keras.models import Sequential, Model
from keras.utils import plot_model
from keras.layers.merge import _Merge
import keras.backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
plt.switch_backend('agg')

class WGANGP():
    
    def __init__(self):
        self.img_row = 28
        self.img_col = 28
        self.channel = 1
        self.img_shape = (self.img_row, self.img_col, self.channel)
        self.latent_dim = 100
        self.n_critic = 5
        
        optimizer = RMSprop(lr=0.00005)

        self.generator = self.build_generator()
        self.critic = self.build_critic()
        self.generator.trainable = False

        real_img = Input(shape=self.img_shape)
        noise = Input(shape=(self.latent_dim,))
        fake_img = self.generator(noise)

        fake = self.critic(fake_img)
        real = self.critic(real_img)

        inter_img = RandomWeightedAverage()([real_img, fake_img])
        val = self.critic(inter_img)

        gp_loss = partial(self.gradient_penalty_loss, averaged_samples=inter_img)
        gp_loss.__name__ = 'gradient_penalty'

        self.critic_model = Model(inputs=[real_img, noise], outputs=[real, fake, val])
        self.critic_model.compile(loss=[self.wasserstein_loss, self.wasserstein_loss, gp_loss], optimizer=optimizer, loss_weights=[1, 1, 10])

        self.critic.trainable = False
        self.generator.trainable = True

        noise = Input(shape=(100,))
        img = self.generator(noise)
        pred = self.critic(img)

        self.generator_model = Model(noise, pred)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, label, pred, averaged_samples):
        
        gradients = K.gradients(pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty)

    def wasserstein_loss(self, label, pred):
        return K.mean(label * pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channel, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()
        plot_model(model, to_file='generator.png')

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()
        plot_model(model, to_file='critic.png')

        img = Input(shape=self.img_shape)
        pred= model(img)

        return Model(img, pred)

    def train(self, epochs, batch_size, sample_interval=50):

        f = np.load('../mnist.npz')
        X_train = f['x_train']
        f.close()

        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        real = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        gp = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            for _ in range(self.n_critic):

                idx = np.random.randint(0, X_train.shape[0], batch_size)
                real_imgs = X_train[idx]
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                d_loss = self.critic_model.train_on_batch([real_imgs, noise], [real, fake, gp])
            g_loss = self.generator_model.train_on_batch(noise, real)

            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=30000, batch_size=32, sample_interval=200)