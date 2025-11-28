############### model parameter
import torch
batchsize = 16
blocksize = 1024
hiddensize = 768
vocabsize = 49152
num_heads = 12
n_layers = 12
device = "cuda" if torch.cuda.is_available() else "cpu"
################ training
learning_rate = 6e-04
warmup_steps = 200
maxstep = 4072*2
dropout = 0