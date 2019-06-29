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
import pickle
from keras.models import model_from_json

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


AM.compile(loss='mse', optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.5), \
            metrics=['accuracy'])
AM.summary()
# load in the data for training
#os.system('mkdir images')
x_train = pd.read_csv('filtered_generator_data.csv', header=0, sep='\t', quotechar='"')
# start small, as this is not optimized yet...
scaler=MinMaxScaler()
train_steps=2000
batch_size=20
save_interval=50
graph_a_loss=[]
graph_d_loss=[]
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
        graph_d_loss.append(d_loss[0])
        y = np.ones([batch_size, 1])
#	for l in self.discriminator.layers:
  #                      weights = l.get_weights()
 #                       weights = [np.clip(w, -0.1, 0.1) for w in weights]
#l.set_weights(weights)
        a_loss = AM.train_on_batch(noise, y)
        graph_a_loss.append(a_loss[0])
        log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
        log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
        print(log_mesg)
#        name="%d_iter.png" % i
 #       if i % save_interval == 0:
#            noise = np.random.uniform(0,real_train.max(),size=[166,100])
 #           name="%d_iter.png" % i
  #          f=G.predict(noise)
   #         test=scaler.inverse_transform(f)
    #  #       t-SNE 
     #       name="%d_iter_tsne.png" % i
            
      #      x=x_train.values
#
       #     sne_stuff=np.concatenate((x,test))
  #          sne_stuff=pd.DataFrame(sne_stuff)
   #         sne_stuff.shape
    #        y=np.ones([2*166,1])
     #       y[166:]=0
      #      sne_stuff['y']=y
       #     sne_stuff['label'] = sne_stuff['y'].apply(lambda k: str(k))
        #    tsne = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=500)
         #   tsne_results = tsne.fit_transform(sne_stuff)
         #   sne_stuff['tsne-2d-one'] = tsne_results[:,0]
          #  sne_stuff['tsne-2d-two'] = tsne_results[:,1]
           # plt.figure(i)
           # plt.savefig(name)
           # sns.scatterplot(
            #    x="tsne-2d-one", y="tsne-2d-two",
            #    hue="y",
            #    palette=sns.color_palette("hls", 2),
            #    data=sne_stuff,
            #    legend="full",
            #    alpha=1
            #)
           # newname="Iteration %d of real vs. generated data" % i
          #  plt.title(newname)
           # plt.savefig(name)
g=np.concatenate((graph_d_loss,graph_a_loss))
g=pd.DataFrame(g)
y=np.array(range(2000))
d=np.array(range(2000))
y=y+1
d=d+1
y=np.concatenate((y,d))
f=np.ones([4000,1])
f[2000:]=0
g['y']=y
g['f']=f
g['0']='one'
plt.figure()
plt.xlabel("yo")
sns.lineplot(
    x="y",y=g[0],hue="f",data=g, alpha=1,legend=0
    )
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over time")
plt.savefig("loss_viz.png")
            # pca
            #newname="%d_iter_PCA.png" % i
            #pca_stuff=np.concatenate((x,test))
            #pca_stuff=pd.DataFrame(pca_stuff)
            #pca = PCA(n_components=3)
            #pca_result = pca.fit_transform(pca_stuff.values)


            #pca_stuff['pca-one'] = pca_result[:,0]
            #pca_stuff['pca-two'] = pca_result[:,1] 
            #pca_stuff['pca-three'] = pca_result[:,2]
            
            
            
            #pca_stuff['y']=y
            #pca_stuff['label'] = pca_stuff['y'].apply(lambda k: str(k))
            #plt.figure(i)

    #        plt.savefig(newname)
            #            plt.figure(i)
     #       sns.scatterplot(
      #          x="pca-one", y="pca-two",
       #         hue="y",
        #        palette=sns.color_palette("hls", 2),
         #       data=pca_stuff,
          #      legend="full",
           #     alpha=1
           # )
        
os.system('mv *.png proj_images')
            
            #t-sne
            
#y[166:]='generated'
            
            
#noise = np.random.uniform(0,real_train.max(),size=[166,100])
           # name="%d_iter.png" % i
#f=G.predict(noise)
#test=scaler.inverse_transform(f)
#test

#concordance_correlation_coefficient(test,x_train.values)
# this one appears to be sufferring from mode collapse --- the same guess every single time.... but maybe thats not a bad thing? definitely is however 



#filename = 'finalized_model_D.sav'
#pickle.dump(D, open(filename, 'wb'))

#filename='finalized_model_G.sav'
model_json = D.to_json()
with open("D.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
D.save_weights("D_weight.h5")

model_json = G.to_json()
with open("G.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
G.save_weights("G_weight.h5")





