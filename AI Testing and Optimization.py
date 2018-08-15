
# coding: utf-8

# # Exploring Optimized and Unoptimized Model Structures

# ### Hyperparameter

# #### No Hyperparameter Optimization

# In[ ]:


model = Sequential()
model.add(Dense(16, activation='linear', input_shape=(784,)))
model.add(BatchNormalization())
model.add(Dropout(0.8))
model.add(Dense(32, activation='linear'))
model.add(BatchNormalization())
model.add(Dropout(0.9))
model.add(Dense(10, activation='softmax'))


# #### Hyperparameter Optimization

# In[ ]:


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(784,)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# ### Model Structure

# #### Underfitting

# In[ ]:


model = Sequential()
model.add(Dense(64, activation = 'sigmoid', input_shape=(784,)))
model.add(Dropout(0.99))
model.add(Dense(10, activation = "softmax"))


# #### Overfitting

# In[ ]:


model = Sequential()
model.add(Dense(64, activation = 'relu', input_shape=(784,)))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(10, activation ='softmax'))

# Epoch 10/10
#60000/60000 [============================] - 124s 2ms/step - loss: 0.6023 - acc:0.8388 - val_loss: 1.0350 - val_acc: 0.6662


# ### Dataset

# #### Image Data Generator

# In[ ]:


batch_size = 16

generator = datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode=None,  
        shuffle=False)  

bottleneck_features_train = model.predict_generator(generator, 2000)

np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)

generator = datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

bottleneck_features_validation = model.predict_generator(generator, 800)


# #### Steps to Preprocess data

# In[ ]:


X_train = X_train.reshape(60000, 784).astype('float32')
X_test = X_test.reshape(10000, 784).astype('float32')


# In[ ]:


X_train /= 255
X_test /= 255


# In[ ]:


n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

