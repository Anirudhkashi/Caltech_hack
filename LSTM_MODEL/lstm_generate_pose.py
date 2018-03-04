
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import utils

import keras
from keras.layers import Input, LSTM, Reshape, Dense, Lambda
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import EarlyStopping


# In[ ]:


n_h = 15

# Input shape (M, T, P)
X_train, Y_train, X_test, Y_test = utils.getGenerativeSequence()
M, Tx, n_values = X_train.shape
Ty = 15


# In[ ]:


reshapor = Reshape((1, n_values))
LSTM_cell_layer1 = LSTM(n_h, activation='relu', return_state=True)
# LSTM_cell_layer2 = LSTM(n_h, return_state = True)
# LSTM_cell_layer3 = LSTM(n_h, return_state = True)
densor = Dense(n_values)
intermmediate_out = Dense(n_h, activation='relu')


def lstmModel1LayerTrain(Tx, Ty, n_h, n_a, n_values):
    
    X_inp = Input(shape=(Tx, n_values))
    
    s0 = Input(shape=(n_h,), name='s0')
    c0 = Input(shape=(n_h,), name='c0')
    s = s0
    c = c0
    
    outputs = []
    
    a = LSTM(6, activation='relu')(X_inp)
    
    for t in range(Ty):
        a = reshapor(a)
        s, _, c = LSTM_cell_layer1(a, initial_state=[s, c])
        out = densor(s)
        outputs.append(out)
        a = out
        
    model = Model([X_inp, s0, c0], outputs)
    
    return model


# In[ ]:

n_a = 15
model = lstmModel1LayerTrain(Tx, Ty, n_h, n_a, n_values)


# In[ ]:


opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])


# In[ ]:


a0 = np.zeros((2, n_h))
c0 = np.zeros((2, n_h))


# In[ ]:


NUM_EPOCHS = 20
BATCH_SIZE = 2

st = 0
end = 2

examples = int(M/2)
for b in range(examples):

    model.fit([X_train[st:end, :, :], a0, c0], list(Y_train[:, st:end, :]), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=[EarlyStopping()])
    st += 2
    end = st + 2


# In[ ]:


st = 0
end = 2

examples = int(X_test.shape[0]/2)
for b in range(examples):
    model.evaluate([X_test[st:end, :, :], a0, c0], list(Y_test[:, st:end, :]), batch_size=BATCH_SIZE)
    st += 2
    end = st + 2

# In[ ]:

model.save_weights("model.h5")
X_predict, file_names = utils.getPredictingData()

out = np.absolute(np.array(model.predict([X_predict, a0, c0], batch_size=2)))

mult = np.ones((15, 1, 6))
mult = mult * 1280

mult[:, :, 1] = np.full((15, 1), 720)
mult[:, :, 3] = np.full((15, 1), 720)
mult[:, :, 5] = np.full((15, 1), 720)

# print out
out = np.multiply(out, mult)
print out

with open("model_out.json", "w") as f:
    f.write(str(out.tolist()))