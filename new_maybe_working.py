# new script 
import statistics
import pandas as pd
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import keras.backend as K
from keras import regularizers
from keras.models import model_from_json

## this particular script will be using the clipping method of wasserstein
## GANs... as opposed to KL divergence metrics

#def wasserstein_loss(y_true, y_pred):
 #       return -K.mean(y_true * y_pred)

# set up the discriminator model....
def mean_confidence_interval(data, confidence=0.95): 
    a = 1.0 * np.array(data) 
    n = len(a) 
    m, se = np.mean(a), scipy.stats.sem(a) 
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1) 
    return m, m-h, m+h, h 
def concordance_correlation_coefficient(y_true, y_pred,
                       sample_weight=None,
                       multioutput='uniform_average'):
    
    cor=np.corrcoef(y_true,y_pred)[0][1]
    
    mean_true=np.mean(y_true)
    mean_pred=np.mean(y_pred)
    
    var_true=np.var(y_true)
    var_pred=np.var(y_pred)
    
    sd_true=np.std(y_true)
    sd_pred=np.std(y_pred)
    
    numerator=2*cor*sd_true*sd_pred
    
    denominator=var_true+var_pred+(mean_true-mean_pred)**2

    return numerator/denominator


def tsne_eval():
    scaler=MinMaxScaler()
    ok=scaler.fit(x_train.values)
    real_train=scaler.transform(x_train.values)
    noise = np.random.uniform(0,real_train.max(),size=[166,100])
           # name="%d_iter.png" % i
    f=G.predict(noise)
    test=scaler.inverse_transform(f)
    x=x_train.values

    sne_stuff=np.concatenate((x,test))
    sne_stuff=pd.DataFrame(sne_stuff)
    sne_stuff.shape
    y=np.ones([2*166,1])
    y[166:]=0
    sne_stuff['y']=y
    sne_stuff['label'] = sne_stuff['y'].apply(lambda k: str(k))
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=2000)
    tsne_results = tsne.fit_transform(sne_stuff)
    sne_stuff['tsne-2d-one'] = tsne_results[:,0]
    sne_stuff['tsne-2d-two'] = tsne_results[:,1]
    plt.figure()
#            plt.savefig(name)
    sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two",
                hue="y",
                palette=sns.color_palette("hls", 2),
                data=sne_stuff,
                legend="full",
                alpha=1
            )
    print("CC:",concordance_correlation_coefficient(x,test))
    
    
x_train = pd.read_csv('filtered_generator_data.csv', header=0, sep='\t', quotechar='"')
# start small, as this is not optimized yet...
def training(train_steps,batch_size,training_ratio):
    scaler=MinMaxScaler()
    train_steps=train_steps
    batch_size=batch_size
    #save_interval=500
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
            if training_ratio > 0:
        
                for j in range(training_ratio):
                    DM.train_on_batch(x,y)
            d_loss = DM.train_on_batch(x,y)
            y = np.ones([batch_size, 1])
#	for l in self.discriminator.layers:
  #                      weights = l.get_weights()
 #                       weights = [np.clip(w, -0.1, 0.1) for w in weights]
#l.set_weights(weights)
            a_loss = AM.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
def create_models(dropout_d,dropout_g,units_d1,units_d2,activation_d,losses,units_g,lr_d,lr_g):     
    D = Sequential()
    dropout=dropout_d
    D.add(Dense(units_d1,input_dim=5000))#,activity_regularizer=regularizers.l2(0.02)))
    D.add(BatchNormalization(momentum=0.8))
    D.add(Dropout(dropout))
    #D.add(BatchNormalization(momentum=0.8))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dense(units_d2))#,activity_regularizer=regularizers.l2(0.02),))          # = this has been changed from 200 to 100!!!!
    D.add(BatchNormalization(momentum=0.8))
    D.add(Dropout(dropout))
    #D.add(BatchNormalization(momentum=0.8))               ### try to add dropout first.. was originally norm,noise,kernel
    D.add(LeakyReLU(alpha=0.2))

    #D.add(Dropout(dropout))
    #D.add(BatchNormalization())
    #D.add(Dense(20,activation='relu'))
    #D.add(BatchNormalization(momentum=0.8))
    D.add(Dense(1,activation=activation_d))
    #D.add(Activation('sigmoid'))
    D.summary()
    DM=Sequential()
    DM.add(D)### this is new addition and analagous to setting the weights of discriminator in AM to not be trainable !!! 
    DM.compile(loss=losses[0],
                  optimizer=SGD(lr=lr_d),
                  metrics=['accuracy'])
    # set up the generator model
    G = Sequential()
    dropout = dropout_g
    G.add(Dense(units_g,input_dim=100))#,activity_regularizer=regularizers.l2(0.02)))#,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
    #G.add(Dropout(dropout)) # add the regularizers again this time with the DM being trained instead.. see if it makes 
    # any difference .... 
    # and maybe keep the trainign ratio this time as the generator appears to be doing well regardless
    G.add(BatchNormalization(momentum=0.8))
    G.add(Dropout(dropout))
    G.add(LeakyReLU(alpha=0.2))
    #G.add(Dense(6000)) # this was added
    #G.add(BatchNormalization(momentum=0.8)) # this was added   ==== = = = = 
    #G.add(Dropout(dropout)) # this was added  = = = = = 
    #G.add(LeakyReLU(alpha=0.2)) # this was added = = = = = = = = 
    G.add(Dense(5000))#,activity_regularizer=regularizers.l2(0.02)))
    #G.add(LeakyReLU(alpha=0.2))
    G.summary()
    # set up the adversarial, wehreby the generators predicted guess will
    # be passed onto the discriminator

    AM = Sequential()
    AM.add(G)
    AM.add(D)


    AM.compile(loss=losses[1], optimizer=Adam(lr=lr_g,beta_1=0.9,beta_2=0.999), \
                metrics=['accuracy'])
    AM.summary()
    return D,G,DM,AM
D,G,DM,AM=create_models(dropout_d=0.6,dropout_g=0.6,lr_d=0.0005,lr_g=0.0002,units_d1=2500,units_d2=400,activation_d='sigmoid', 
                           losses=['binary_crossentropy','binary_crossentropy'],units_g=2000)
training(25000,40,0)
tsne_eval()
plt.savefig("viz_for_model.png")
tsne_eval()
plt.savefig("second_look.png")
