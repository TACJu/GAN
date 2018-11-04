import numpy as np
import matplotlib.pyplot as plt
import os

from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, ZeroPadding2D, Dropout, BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from keras.models import Sequential, Model
from keras.utils import plot_model
import keras.backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
plt.switch_backend('agg')

class WGAN():
    
    def __init__(self):
        self.row = 28
        self.col = 28
        self.channel = 1
        self.img_shape = (self.row, self.col, self.channel)
        self.latent_dim = 100
        self.n_critic = 5
        self.clip_value = 0.01

        optimizer = RMSprop(lr = 0.00005)

        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])
        self.critic.trainable = False

        self.generator = self.build_generator()

        noise = Input(shape=(self.latent_dim,))
        img = self.generator(noise)

        pred = self.critic(img)

        self.combined = Model(noise, pred)
        self.combined.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])

    def wasserstein_loss(self, label, pred):
        return K.mean(label * pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channel, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()
        plot_model(model, to_file='generator.png')

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()
        plot_model(model, to_file='critic.png')

        img = Input(shape=self.img_shape)
        pred = model(img)

        return Model(img, pred)

    def train(self, epochs, batch_size=64, save_interval=50):

        f = np.load('../mnist.npz')
        X_train = f['x_train']
        f.close()

        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        real = -np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            for i in range(self.n_critic):
            
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                real_imgs = X_train[idx]

                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                fake_imgs = self.generator.predict(noise)

                d_loss_real = self.critic.train_on_batch(real_imgs, real)
                d_loss_fake = self.critic.train_on_batch(fake_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                for layer in self.critic.layers:
                    weights = layer.get_weights()
                    weights = [np.clip(weight, -self.clip_value, self.clip_value) for weight in weights]
                    layer.set_weights(weights)

            g_loss = self.combined.train_on_batch(noise, real)

            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
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


if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=30000, batch_size=64, save_interval=200)
