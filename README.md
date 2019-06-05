# thesis

As of version 3, the code appears to be running and moving along a gradient towards a minimum error estimate... the initial versions had the problem of vanishing or no gradient loss at all, but the problem here appeared to be the scaling. When the units have been scaled for variance and mean, the model appears to be working, whereby the discriminator finds it increasingly more difficult to tell real from fake. The issue now is the scaling back (as of now, i think this should involve minmaxscaler().inverse_transform()) of the end product, if successful, to try and interpret the results. The question then also remains; how to quantify the bounds of what the model has essentially captured? 


#### edit - this actually worked and performed quite well... but the gradient eventually disappeared unfortunately. So as of now, running for approximately 1500 epochs yielded positive results, whereby the discriminator was unable to tell the difference between the real and the fake data.... very promising! Must make a graph of the performance??? 
