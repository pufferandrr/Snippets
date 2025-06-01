import torch
import numpy as np

class argparse():
    pass

args = argparse()
args.epochs, args.learning_rate, args.patience = [30, 0.001, 4]
args.hidden_size, args.input_size= [40, 30]
args.device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),]