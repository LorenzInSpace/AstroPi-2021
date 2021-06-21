import torch
import torch.nn as nn
import torch.nn.functional as F

# Questi sono esempi di come si creano i layer per una rete neurale, applicando funzioni di attivazione, come ReLU e la sigmoide


# option 1 (create nn modules)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size): # rete neurale con layer nascosto, il primo layer ha dimensione input_size, quello nascosto hidden_size
        # va usato per forza il costruttore di nn.Module
        super(NeuralNet, self).__init__()
        
        # Questo è il primo layer, prima applica y = w*x + b all'input, poi vi applica una ReLU (rectified linear unit, in pratica y = x se x > 0, 0 altrimenti
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()     # oppure nn.Sigmoid, nn.Softmax, nn.TanH, nn.LeakyReLU
        
        # Questo è l'hidden layer, applica y = w*x + b come prima, e poi una sigmoide
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
  
    # si scrive esplicitamente come calcolare l'output
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


# option 2 (use activation functions directly in forward pass)
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        #torch.Sigmoid, torch.Softmax, torch.TanH
        #if a function is not available in torch, it is in F, like Leaky Relu
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out
        
