
# coding: utf-8

# # Vanilla LSTM in Keras

# In this notebook, we use an LSTM to classify IMDB movie reviews by their sentiment.

# #### Load dependencies

# In[5]:


import keras
from keras.layers import Dense, Dropout, Embedding, SpatialDropout1D, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
import matplotlib.pyplot as plt


# #### Load data

# In[10]:


(X_train, y_train), (X_valid, y_valid) = imdb.load_data(num_words=8000)


# #### Preprocess data

# In[11]:


X_train = pad_sequences(X_train, maxlen=125, padding='pre', truncating='pre', value=0)
X_valid = pad_sequences(X_valid, maxlen=125, padding='pre', truncating='pre', value=0)


# #### Design neural network architecture

# In[12]:


model = Sequential()
model.add(Embedding(8000, 50, input_length=125))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(256, dropout=0.25))
model.add(Dense(1, activation='sigmoid'))


# In[13]:


model.summary()


# #### Configure model

# In[14]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# #### Train!

# In[16]:


model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, validation_data=(X_valid, y_valid))
