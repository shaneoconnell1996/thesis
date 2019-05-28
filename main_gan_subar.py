'''
The following is a script that carries out a generative adversarial network on kidney transplant gene expression data to classify whether or not they will go on to reject their graft transplant. 
'''
import statistics
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " seconds"
        elif sec < (60 * 60):
            return str(sec / 60) + " minutes"
        else:
            return str(sec / (60 * 60)) + " hours"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )


class GAN(object):
    def __init__(self,row_samples=166,col_genes=5000): ##There is already a problem here. The input space considered here may not be exactly this, owing to the fact that there will be a split performed later with scikit. Hence this needs to be rewritten to perhaps not contain the number of samples. Unless of course the split can come here *using* this number specified, inwhich case givingthe total number of samples is not a problem.
        
        self.row_samples=row_samples
        self.col_genes=col_genes
        self.D = None   # first discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None # discriminator model
        
       
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential() ## this is the start of the discriminator model, for which we will start by adding just one layer
        #depth=659 ## for now we will consider this space as we would consider an image... so here, the number of rows correspond to the 3rd dimension of the space.
        ## next, how to derive the optimum number of dropouts? The dropout rate will prevent the model from overfitting to the data ... we can initially set this quite harsh, at about 0.4 
        dropout=0.3 ## here we will deviate from the strict code - this is because we will use some of the code used for a different 
        self.D.add(Dense(round(0.5*self.col_genes),input_shape=(self.col_genes,)))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
                   ## the more complex activation functions here have to be added as their own layers. this leaky relu will not discard negative values which may have a contribution to the output
        # we will also start for now by just applying one hidden layer - we want this to generalise well to new data
        ## the dropout here will ensure overfitting - also need to consider the option of pruning the nodes in the network -which is to say, an optimisation technique for seeing what nodes are non-essential. 
                   
                   
                   
        ## output layer
                   
       # self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D
        ## this is very important to note - when you are adding a dense layer to a neural network, the first command specifies the first 2 layers in reaity, those bing the input shape and the first hidden layer - which explains some of the more confusing numbers rigth at the start. For instance, this will be showcased in the design of the generator matrix.            
    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.3
        self.G.add(Dense(int(self.col_genes/2),input_dim=100))
        #self.G.add(BatchNormalization(momentum=0.8))
        self.G.add(Activation('relu'))
        #self.G.add(Reshape(int(self.row_samples/2),int(2*self.col_genes)))
        self.G.add(Dropout(dropout))
                   
                   
        
        self.G.add(Dense(self.col_genes))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G
        
                   
                   
                   
                   # this appears to signal the start of the next layer of convolution, but we must go in the opposite direction to what we would normally go at. Instead of dimensionality reduction , we need to multipy through a transpose command to maintain the same relationships from the input layer onto the hidden layer. 
                   
                   # The next chunk is important, as it is intialisation of the final model. I woudl personally like to, from my own experience and reading, implement a mean absolute error metric for backpropagation rather than the usual RMSE. As well, because I am more familiar with adam, I will use that instead of RMSProp 
                   
    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
	self.DM.summary()
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='mean_absolute_error', optimizer=optimizer,\
            metrics=['accuracy'])
	self.AM.summary()
        return self.AM
                   
                   
class GAN_perform(object):
    def __init__(self):
        

        self.x_train = pd.read_csv('filtered_generator_data.csv', header=0, sep='\t', quotechar='"')

        self.GAN = GAN()
        self.discriminator =  self.GAN.discriminator_model()
        self.adversarial = self.GAN.adversarial_model()
        self.generator = self.GAN.generator()

    def train(self, train_steps=20000, batch_size=100):
        noise_input = None
#        if save_interval>0:
 #           noise_input = np.random.uniform(1.0, 15.0, size=[166, 200])
        for i in range(train_steps):
            real_train = self.x_train.values[np.random.randint(0,self.x_train.shape[0], size=batch_size)]
            noise = np.random.uniform(1.0, 15.0, size=[batch_size, 100])
            fake_train = self.generator.predict(noise)
            x = np.concatenate((real_train, fake_train))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(1.0, 15.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            
### summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
    
 # all of this code will be good once the history has been accessed, and can then be used to plot the accuracy versus the epochs                   
         
                
                 
if __name__ == '__main__':
    gan = GAN_perform()
    timer = ElapsedTimer()
    gan.train(train_steps=20000, batch_size=100)
    timer.elapsed_time()                   

        







