############### model parameter
import torch
batchsize = 16
blocksize = 16
hiddensize = 12
vocabsize = 49152
num_heads = 4
n_layers = 12
device = "cuda" if torch.cuda.is_available() else "cpu"
################ training
learning_rate = 6e-04
warmup_steps = 200
maxstep = 4072*2
dropout = 0