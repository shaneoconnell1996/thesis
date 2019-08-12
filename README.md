## Thesis code breakdown 

The following set of scripts details the efforts of my 10 week long MSc project completed in fulfilment of my degree in Biomedical Genomics, whereby a Generative Adversarial Network was applied to kidney transplant rejection data to model p(x,y), which is the joint probability distribution of the features associated with the label (graft loss). Numbered accordingly, the scripts indicate the progression of model evolutions towards the final working algorithm, which achieved good performance on validation metrics in concordance correlation and dimensionality reduction clustering. Wasserstein approaches were not implemented, but the gradient normalisation flag built in to Keras was used for the final models. Any queries can be directed to shane.connell96@gmail.com.
#### Dependencies: 
Python 3.6
Tensorflow 1.10.0
