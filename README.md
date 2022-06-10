# AMD_demo
This is a proof of concept script of the paper https://www.nature.com/articles/s41597-020-0473-z
The original paper is based on ASE_ANI, but here we use torchani instead
For demo purpose here we only implemented the MD sampling described in the paper

pool: 
path to store initial dataset, and new data sampled during the AL loop

sampler.py: 
Wrapper functions to run MD sampling, and QM calculations after getting new structures
Using ASE package to run MD with pre-trained ANI potential
Using ORCA5 to run QM calculation 

trainer.py: 
Function to train the model once new structures are generated during each AL epoch
Different from the original ANI paper, we use LAMB optimizer here

utils.py: 
helper functions 
