# Goal
This repository contains my work of using variational auto encoder (VAE) to solve class imbalance problem.

# Problem
We want to build a classifier to differentiate images of digits 0 from images of digits 6. We have 50000 images of 0 and only 100 images of 6. 

# Solution
Train a VAE on 100 images of 6 and then generate more images of 6 from this trained VAE. Use these extra images for training the classifier.

# Result
The VAE based data augmentation approach gives 50% improvement in accuracy compare to oversampling based approach to overcome the class imbalance problem. 

# Reason
VAE was able to produce more general images such as different rotation of number 6 along with different thickness. Thus the classification network was able to use these images to generalize over the test data set.

# Use

To generate additional images of 6, run the following code

`cd vae`
`python main.py`

This will generate a data file named `vae_samples.npy` which contains additional images of digit 6.

To train the model without VAE based data augmentation, run the following code

`python mnist_train.py --training_type without_vae`

To train the model with VAE based data augmentation, run the following code

`python mnist_train.py --training_type with_vae`

