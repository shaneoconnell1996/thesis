import statistics
import pandas as pd
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import keras.backend as K
from keras import regularizers
## this particular script will be using the clipping method of wasserstein
## GANs... as opposed to KL divergence metrics

#def wasserstein_loss(y_true, y_pred):
 #       return -K.mean(y_true * y_pred)

# set up the discriminator model....
D = Sequential()
dropout=0.6
D.add(Dense(2000,input_dim=5000))
D.add(BatchNormalization(momentum=0.8))
D.add(Dropout(dropout))
D.add(LeakyReLU(alpha=0.2))


D.add(Dense(200))
D.add(BatchNormalization(momentum=0.8))
D.add(Dropout(dropout))
D.add(LeakyReLU(alpha=0.2))

#D.add(Dropout(dropout))
#D.add(BatchNormalization())
#D.add(Dense(20,activation='relu'))
#D.add(BatchNormalization(momentum=0.8))
D.add(Dense(1,activation='sigmoid'))
#D.add(Activation('sigmoid'))
D.summary()
D.compile(loss='mse',
              optimizer='sgd',
              metrics=['accuracy'])
# set up the generator model
G = Sequential()
dropout = 0.6
G.add(Dense(2000,input_dim=100))#,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
G.add(BatchNormalization(momentum=0.8))
G.add(Dropout(dropout))
G.add(LeakyReLU(alpha=0.2))
G.add(Dense(4000,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
#G.add(BatchNormalization(momentum=0.8))
#G.add(Activation('relu'))
        #self.G.add(Reshape(int(self.row_samples/2),int(2*self.col_genes)))
#G.add(Dropout(dropout))
#G.add(Dense(4000))
#G.add(LeakyReLU(alpha=0.2))
#G.add(BatchNormalization(momentum=0.8))
G.add(Dense(5000))#,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
#G.add(LeakyReLU(alpha=0.2))
G.summary()
# set up the adversarial, wehreby the generators predicted guess will
# be passed onto the discriminator

AM = Sequential()
AM.add(G)
AM.add(D)


AM.compile(loss='mse', optimizer=Adam(lr=0.0005,beta_1=0.9,beta_2=0.5), \
            metrics=['accuracy'])
AM.summary()
# load in the data for training
#os.system('mkdir images')
x_train = pd.read_csv('filtered_generator_data.csv', header=0, sep='\t', quotechar='"')
# start small, as this is not optimized yet...
scaler=MinMaxScaler()
#train_steps=20000
batch_size=20
save_interval=500


for i in range(train_steps):
        real_train = x_train.values[np.random.randint(0, \
166,size=batch_size)] # take a random subset of the data
        ok=scaler.fit(real_train)
        real_train=scaler.transform(real_train)
        noise = np.random.uniform(0.0, real_train.max(), size=[batch_size, 100])
            #noise = tf.convert_to_tensor(noise,dtype=tf.float32)
        fake_train = G.predict(noise)
            #tf.Session().run(fake_train)
        x = np.concatenate((real_train, fake_train))
        y = np.ones([2*batch_size, 1])
        y[batch_size:, :] = 0
        #d_loss = DM.train_on_batch(x, y)
#        if i % 10 == 0:

 #           for l in D.layers:
  #                          weights = l.get_weights()
        d_loss = D.train_on_batch(x,y)
        y = np.ones([batch_size, 1])
#       for l in self.discriminator.layers:
  #                      weights = l.get_weights()
 #                       weights = [np.clip(w, -0.1, 0.1) for w in weights]
#l.set_weights(weights)
        a_loss = AM.train_on_batch(noise, y)
        log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
        log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
        print(log_mesg)


noise = np.random.uniform(0,real_train.max(),size=[166,100])
           # name="%d_iter.png" % i
f=G.predict(noise)
test=scaler.inverse_transform(f)
test

concordance_correlation_coefficient(test,x_train.values)
# this one appears to be sufferring from mode collapse --- the same guess every single time.... but maybe thats not a bad thing? definitely is however 

## need to add the in house functions for CC ...






