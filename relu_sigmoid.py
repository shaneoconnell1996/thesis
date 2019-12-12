# new script 
import statistics
import pandas as pd
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
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
# performance metrics using Lin's Correlation Coefficient

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

# function for creating tsne plot
def tsne_eval():
    scaler=MinMaxScaler()
    ok=scaler.fit(x_train.values) # just to ensure the stdout message from fitting does not go to log file 
    real_train=scaler.transform(x_train.values)
    noise = np.random.uniform(0,real_train.max(),size=[166,100]) # uniform prior 
           # name="%d_iter.png" % i
    f=G.predict(noise) # G(z) predicted
    test=scaler.inverse_transform(f) # inverse transform to get G(z) in workable terms
    x=x_train.values # real values

    sne_stuff=np.concatenate((x,test))
    sne_stuff=pd.DataFrame(sne_stuff)
    #sne_stuff.shape
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
    # same process for PCA Plot
def pca_eval():
    scaler=MinMaxScaler()
    ok=scaler.fit(x_train.values)
    real_train=scaler.transform(x_train.values)
    noise = np.random.uniform(0,real_train.max(),size=[166,100])
           # name="%d_iter.png" % i
    f=G.predict(noise)
    test=scaler.inverse_transform(f)
    x=x_train.values

    y=np.ones([2*166,1])
    y[166:]=0
    pca_stuff=np.concatenate((x,test))
    pca_stuff=pd.DataFrame(pca_stuff)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(pca_stuff.values)


    pca_stuff['pca-one'] = pca_result[:,0]
    pca_stuff['pca-two'] = pca_result[:,1] 
    pca_stuff['pca-three'] = pca_result[:,2]



    pca_stuff['y']=y
    pca_stuff['label'] = pca_stuff['y'].apply(lambda k: str(k))
    #plt.figure(i)

    #plt.savefig(newname)
    plt.figure()
    sns.scatterplot(x="pca-one", y="pca-two",
                    hue="y",
                    palette=sns.color_palette("hls", 2),
                    data=pca_stuff,
                    legend="full",
                    alpha=1)
       # read in the actual data here 
x_train = pd.read_csv('filtered_generator_data.csv', header=0, sep='\t', quotechar='"')
# Create training function, that will allow for quick and easy hyperparameter changing 
def training(train_steps,batch_size,training_ratio,save_interval):
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
            
            
            fake_train = G.predict(noise)

            x = np.concatenate((real_train, fake_train))
            
            y = np.ones([2*batch_size, 1])
            
            y[batch_size:, :] = 0
            
            if training_ratio > 0:
        
                for j in range(training_ratio):
                    DM.train_on_batch(x,y)
            d_loss = DM.train_on_batch(x,y)
            y = np.ones([batch_size, 1])
            # naive weight clipping implementeed here, but didnt work so is nothing fancy.... 
#	for l in self.discriminator.layers:
  #                      weights = l.get_weights()
 #                       weights = [np.clip(w, -0.1, 0.1) for w in weights]
#l.set_weights(weights)
            a_loss = AM.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if i % save_interval == 0:
                tsne_eval()
def create_models(dropout_d,dropout_g,units_d1,units_d2,activation_d,losses,units_g,lr_d,lr_g):     
    D = Sequential()
    dropout=dropout_d
    D.add(Dense(units_d1,input_dim=5000,activation='relu'))#,activity_regularizer=regularizers.l2(0.02)))
    D.add(BatchNormalization(momentum=0.8))
    D.add(Dropout(dropout))
    #D.add(BatchNormalization(momentum=0.8))
    #D.add(LeakyReLU(alpha=0.2))
    D.add(Dense(units_d2,activation='relu'))#,activity_regularizer=regularizers.l2(0.02),))          # = this has been changed from 200 to 100!!!!
    D.add(BatchNormalization(momentum=0.8))
    D.add(Dropout(dropout))
    #D.add(BatchNormalization(momentum=0.8))               ### try to add dropout first.. was originally norm,noise,kernel
    #D.add(LeakyReLU(alpha=0.2))

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
    G.add(Dense(units_g,input_dim=100,activation='relu'))#,activity_regularizer=regularizers.l2(0.02)))#,kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
    #G.add(Dropout(dropout)) # add the regularizers again this time with the DM being trained instead.. see if it makes 
    # any difference .... 
    # and maybe keep the trainign ratio this time as the generator appears to be doing well regardless
    G.add(BatchNormalization(momentum=0.8))
    G.add(Dropout(dropout))
    #G.add(LeakyReLU(alpha=0.2))
    G.add(Dense(units_g,activation='relu')) # this was added
    G.add(BatchNormalization(momentum=0.8)) # this was added   ==== = = = = 
    G.add(Dropout(dropout)) # this was added  = = = = = 
    #G.add(LeakyReLU(alpha=0.2)) # this was added = = = = = = = = 
    G.add(Dense(5000,activation='relu'))#,activity_regularizer=regularizers.l1(0.02)))
    #G.add(LeakyReLU(alpha=0.2))
    G.summary()
    # set up the adversarial, wehreby the generators predicted guess will
    # be passed onto the discriminator

    AM = Sequential()
    AM.add(G)
    AM.add(D)


    AM.compile(loss=losses[1], optimizer=Adam(lr=lr_g,beta_1=0.9,beta_2=0.999,clipnorm=1), \
                metrics=['accuracy'])
    AM.summary()
    return D,G,DM,AM
D,G,DM,AM=create_models(dropout_d=0.3,dropout_g=0.3,lr_d=0.0002,lr_g=0.0001,units_d1=300,units_d2=300,activation_d='sigmoid', 
                           losses=['binary_crossentropy','binary_crossentropy'],units_g=3000)
training(20000,65,0,2000)
tsne_eval()
plt.savefig("relu_sig.png")
tsne_eval()
plt.savefig("relu_sig_2.png")
pca_eval()
plt.savefig("relu_sig_PCA.png")

