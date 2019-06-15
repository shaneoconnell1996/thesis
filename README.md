# thesis

As of version 3, the code appears to be running and moving along a gradient towards a minimum error estimate... the initial versions had the problem of vanishing or no gradient loss at all, but the problem here appeared to be the scaling. When the units have been scaled for variance and mean, the model appears to be working, whereby the discriminator finds it increasingly more difficult to tell real from fake. The issue now is the scaling back (as of now, i think this should involve minmaxscaler().inverse_transform()) of the end product, if successful, to try and interpret the results. The question then also remains; how to quantify the bounds of what the model has essentially captured? 


#### edit - this actually worked and performed quite well... but the gradient eventually disappeared unfortunately. So as of now, running for approximately 1500 epochs yielded positive results, whereby the discriminator was unable to tell the difference between the real and the fake data.... very promising! Must make a graph of the performance??? 


#### edit for version 5 --- subsequent attempts to stabilise the performance of the gan have not been successful.... discriminator converges on 50 percent accuracy very quickly.... maybe need to add some extra variation in there... ie, different sources of variation, or perhaps a more complete dataset???? could include more genes as this is currently running comfortably in about 90 mins




#### edit for version 6 --- The GAN is trainign correctly but appears to be suffering from mode collapse... the initial PCA plots seem to suggest that the data generated from the noise does not cover the full extent of variation present in the real dataset .... need to try and tweak the model to fix this... 



#### edit for version 7 as of 15th of June... 
The mode collapse could potentoally be resolved by adding a different penalty term - the WGAN-GP, wherein a 1-Lipschitz constraint is imposed upon the gradient descent. This can ensure stable training, but trying to implement this in the current workflow is proving challenging. IN addition, need to look at some datasets where it is known there is some variation - great to talk about in the thesis. 
