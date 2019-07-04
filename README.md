# thesis

As of version 3, the code appears to be running and moving along a gradient towards a minimum error estimate... the initial versions had the problem of vanishing or no gradient loss at all, but the problem here appeared to be the scaling. When the units have been scaled for variance and mean, the model appears to be working, whereby the discriminator finds it increasingly more difficult to tell real from fake. The issue now is the scaling back (as of now, i think this should involve minmaxscaler().inverse_transform()) of the end product, if successful, to try and interpret the results. The question then also remains; how to quantify the bounds of what the model has essentially captured? 


#### edit - this actually worked and performed quite well... but the gradient eventually disappeared unfortunately. So as of now, running for approximately 1500 epochs yielded positive results, whereby the discriminator was unable to tell the difference between the real and the fake data.... very promising! Must make a graph of the performance??? 


#### edit for version 5 --- subsequent attempts to stabilise the performance of the gan have not been successful.... discriminator converges on 50 percent accuracy very quickly.... maybe need to add some extra variation in there... ie, different sources of variation, or perhaps a more complete dataset???? could include more genes as this is currently running comfortably in about 90 mins




#### edit for version 6 --- The GAN is trainign correctly but appears to be suffering from mode collapse... the initial PCA plots seem to suggest that the data generated from the noise does not cover the full extent of variation present in the real dataset .... need to try and tweak the model to fix this... 



#### edit for version 7 as of 15th of June... 
The mode collapse could potentoally be resolved by adding a different penalty term - the WGAN-GP, wherein a 1-Lipschitz constraint is imposed upon the gradient descent. This can ensure stable training, but trying to implement this in the current workflow is proving challenging. IN addition, need to look at some datasets where it is known there is some variation - great to talk about in the thesis. 


## edit for version 8 - 23rd of June...

Now appears to be working without the wasserstein loss function ... but in much less epochs. However, both networks are quite beefy, and have 2 hefty hidden layers each. The next step now after the evaluation with concordance correlation (which performed quite well) is to extract the nodes at the top layer of the generator and see what is going on... can they be interpreted in a meaningful fashion? Very interesting... the idea is to go back to original analysis and see if any of the gene sets identified to be significant to kidney transplant are thrown up in the weights of the network after some ranking method ... this will be true proof of principle... there is also the matter of the tsne plot not looking exactly correct just yet --- shouldnt be pulled apart the way it has been. Maybe try PCA? or am I going to trust the concordance correlation metrics? After this the downstream application will involve something like the predictor built on unsupervised cluster assignment. 




## edit for version 9 -
Tried adding a leaky relu activation function at the top layer instead of the linear one that had been used before .... seems to achieve the same results - the error term never quite reaches the minimum needed to actually imitate the data like in the sparse network. 

## most recent model -- has not undergone full evaluation cycle
Have vastly imporved the custimizability of the network allowing for quicker validation assessments. These paramaters below have not undergone full testing but seem to have some promise with evidence of less severe mode collapse via t-sne plots. 

D,G,DM,AM=create_models(dropout_d=0.5,dropout_g=0.5,lr_d=0.0002,lr_g=0.0002,units_d1=2500,units_d2=400,activation_d='sigmoid', 
                           losses=['binary_crossentropy','binary_crossentropy'],units_g=2000)]
training(9000,60,0,2000) <- this is where I think there is a lot of promise ... by upping the batch size, this appears to be working a little bit better. Although of course it would be nece to have a generalizable model that can take very small batches, I think a higher batch number seems to be increasing the performance overall.  Also, important to note is the switch from MSE loss metrics to binary crossentropy. This is the recommended loss metric for many of the GAN papers I have read, and also appears to be working quite well. Have also tried to customize the learning rate.. making it smaller seems to help avoid local minima which was a problem with the initial parameters. 
                           
