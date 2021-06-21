import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# Stesso DataSet di prima, ma qui si provano le trasformate, delle funzioni che vengono applicate al DataSet in modo da velocizzare alcune cose,
# tipo il non dover convertire da Numpy a Torch

class WineDataset(Dataset):

    def __init__(self, transform= None):
        #data loading
        xy = np.loadtxt('./Dati/wine.csv', delimiter= ",", dtype= np.float32, skiprows= 1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]] # n_samples, 1
        self.n_samples = xy.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self):
        return self.n_samples


# si crea una trasformata personalizzata, quando viene chiamata restituisce i dati da numpy a torch
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

# altra trasformata personalizzata, moltiplica i dati per uno stesso fattore
class MulTransform:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor

        return inputs, target


dataset = WineDataset(transform = ToTensor())

first_data = dataset[0]
features, labels = first_data
print(features, labels)

# crea una trasformata che è la composizione delle prime due
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineDataset(transform= composed)

second_data = dataset[0]
features, labels = second_data
print(features, labels)
