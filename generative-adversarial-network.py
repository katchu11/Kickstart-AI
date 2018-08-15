
# coding: utf-8

# # GAN for Quick Draw Dataset

# * code based directly on [Grant Beyleveld's](https://github.com/grantbey/quickdraw-GAN/blob/master/octopus-v1.0.ipynb), which is derived from [Rowel Atienza's](https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0) under [MIT License](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/LICENSE)
# * data provided by [Google](https://github.com/googlecreativelab/quickdraw-dataset) under [Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/)

# #### Load dependencies

# In[3]:


import numpy as np
import os

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, Flatten, Activation, Reshape, Conv2DTranspose, UpSampling2D
from keras.optimizers import Adam

import pandas as pd
from matplotlib import pyplot as plt


# #### Load data

# In[8]:


input_images = "apple.npy"


# In[9]:


data = np.load(input_images)


# In[10]:


data.shape


# In[11]:


data[4242]


# In[12]:


data = data/255
data = np.reshape(data,(data.shape[0],28,28,1)) # 4th dim is color
img_w,img_h = data.shape[1:3]
data.shape


# In[14]:


plt.imshow(data[20,:,:,0], cmap='Greys')


# #### Create discriminator network

# In[20]:


def discriminator_builder(depth=64,p=0.4):

    inputs = Input((img_w,img_h,1))

    conv1 = Conv2D(depth*1, 5, strides=2, padding='same', activation='relu')(inputs)
    conv1 = Dropout(p)(conv1)

    conv2 = Conv2D(depth*2, 5, strides=2, padding='same', activation='relu')(conv1)
    conv2 = Dropout(p)(conv2)

    conv3 = Conv2D(depth*4, 5, strides=2, padding='same', activation='relu')(conv2)
    conv3 = Dropout(p)(conv3)

    conv4 = Conv2D(depth*8, 5, strides=1, padding='same', activation='relu')(conv3)
    conv4 = Flatten()(Dropout(p)(conv4))

    output = Dense(1, activation='sigmoid')(conv4)

    model = Model(inputs=inputs, outputs=output)
    model.summary()

    return model


# In[21]:


discriminator = discriminator_builder()


# In[22]:


discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.0008, decay=6e-8, clipvalue=1.0),
                      metrics=['accuracy'])


# #### Create generator network

# In[23]:


def generator_builder(z_dim=100,depth=64,p=0.4):

    inputs = Input((z_dim,))

    dense1 = Dense(7*7*64)(inputs)
    dense1 = BatchNormalization(momentum=0.9)(dense1)
    dense1 = Activation(activation='relu')(dense1)
    dense1 = Reshape((7,7,64))(dense1)
    dense1 = Dropout(p)(dense1)

    conv1 = UpSampling2D()(dense1)
    conv1 = Conv2DTranspose(int(depth/2), kernel_size=5, padding='same', activation=None,)(conv1)
    conv1 = BatchNormalization(momentum=0.9)(conv1)
    conv1 = Activation(activation='relu')(conv1)

    conv2 = UpSampling2D()(conv1)
    conv2 = Conv2DTranspose(int(depth/4), kernel_size=5, padding='same', activation=None,)(conv2)
    conv2 = BatchNormalization(momentum=0.9)(conv2)
    conv2 = Activation(activation='relu')(conv2)

    conv3 = Conv2DTranspose(int(depth/8), kernel_size=5, padding='same', activation=None,)(conv2)
    conv3 = BatchNormalization(momentum=0.9)(conv3)
    conv3 = Activation(activation='relu')(conv3)

    output = Conv2D(1, kernel_size=5, padding='same', activation='sigmoid')(conv3)

    model = Model(inputs=inputs, outputs=output)
    model.summary()

    return model


# In[24]:


generator = generator_builder()


# #### Create adversarial network

# In[25]:


def adversarial_builder(z_dim=100):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.0004, decay=3e-8, clipvalue=1.0),
                  metrics=['accuracy'])
    model.summary()
    return model


# In[26]:


adversarial_model = adversarial_builder()


# #### Train!

# In[27]:


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


# In[28]:


def train(epochs=2000,batch=128):

    d_metrics = []
    a_metrics = []

    running_d_loss = 0
    running_d_acc = 0
    running_a_loss = 0
    running_a_acc = 0

    for i in range(epochs):

        if i%100 == 0:
            print(i)

        real_imgs = np.reshape(data[np.random.choice(data.shape[0],batch,replace=False)],(batch,28,28,1))
        fake_imgs = generator.predict(np.random.uniform(-1.0, 1.0, size=[batch, 100]))

        x = np.concatenate((real_imgs,fake_imgs))
        y = np.ones([2*batch,1])
        y[batch:,:] = 0

        make_trainable(discriminator, True)

        d_metrics.append(discriminator.train_on_batch(x,y))
        running_d_loss += d_metrics[-1][0]
        running_d_acc += d_metrics[-1][1]

        make_trainable(discriminator, False)

        noise = np.random.uniform(-1.0, 1.0, size=[batch, 100])
        y = np.ones([batch,1])

        a_metrics.append(adversarial_model.train_on_batch(noise,y))
        running_a_loss += a_metrics[-1][0]
        running_a_acc += a_metrics[-1][1]

        if (i+1)%500 == 0:

            print('Epoch #{}'.format(i+1))
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, running_d_loss/i, running_d_acc/i)
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, running_a_loss/i, running_a_acc/i)
            print(log_mesg)

            noise = np.random.uniform(-1.0, 1.0, size=[16, 100])
            gen_imgs = generator.predict(noise)

            plt.figure(figsize=(5,5))

            for k in range(gen_imgs.shape[0]):
                plt.subplot(4, 4, k+1)
                plt.imshow(gen_imgs[k, :, :, 0], cmap='gray')
                plt.axis('off')

            plt.tight_layout()
            plt.show()

    return a_metrics, d_metrics


# In[19]:


a_metrics_complete, d_metrics_complete = train(epochs=3000)


# In[22]:


ax = pd.DataFrame(
    {
        'Generator': [metric[0] for metric in a_metrics_complete],
        'Discriminator': [metric[0] for metric in d_metrics_complete],
    }
).plot(title='Training Loss', logy=True)
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")


# In[23]:


ax = pd.DataFrame(
    {
        'Generator': [metric[1] for metric in a_metrics_complete],
        'Discriminator': [metric[1] for metric in d_metrics_complete],
    }
).plot(title='Training Accuracy')
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")
