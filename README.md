# Conditional Image Generation for the Fashion Industry

This document is intended to give an overview of the code that has been developed during the thesis. I highlight that some of the files are extrated or based on the repository https://github.com/rosinality/stylegan2-pytorch.
In order to reproduce the results of the thesis, the dataset and the pre-trained weights are made available in the folder https://drive.google.com/drive/folders/1JfJ53vAmeU5KA6CI2iD8zD8lBnlFiU-G?usp=sharing.

## Training
In order to train the model, the dataset needs to be prepared through the script *prepare_data.py* and then, the main script *train.py* starts the training routine. 

## Conditional generation
After the model reaches a satisfactory result, the *space_exploring* files are used to create conditional content. They mainly receive in input the latent code representations of the input images (obtained through the *projector.py* script) and the cehckpoint of the generator network (obtained from the previous step). 
In order to run the second methodology, the intermediate mapping network needs to be trained first. In this case, teh script *sample_mapping_network.py* is used.  