import torch
import torch.nn as nn
import numpy as np

#Si giochicchia con le funzioni softmax e crossEntropy per capirle meglio

# softmax è tipo la funzione logistica, dati degli input X_i, softmax(X_0) = e^(X_0) / sum(e^(X_i))
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# output con il softmax nostro
x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print("softmax numpy: ", outputs)

# output con il softmax di torch
x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim= 0)
print(outputs)





# è tipo l'entropia di Shannon
def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

# y indica come deve essere l'output, quindi quale tra le varie opzioni è quella giusta
# y must be one hot encoded (all values 0 except one, which has 1)
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
Y = np.array([1, 0, 0]) # la classe giusta è la 1

# y_pred has probabilities
Y_pred_good = np.array([0.7, 0.2, 0.1]) # ha previsto che la classe è giusta è la 1, vero
Y_pred_bad = np.array([0.1, 0.3, 0.6]) # previsto che la classe giusta è la 3, falso
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)

print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')



# !!! nn.CrossEntropyLoss applies nn.LogSoftmax + nn.NLLLoss, so NO NEED OF SOFTMAX IN LAST LAYER
# Y has class labels, not One-Hot!
# Y_pred has raw scores (logits), no Softmax!

loss = nn.CrossEntropyLoss()

#3 samples
# Tra 3 possibili classi, Y[i] indica quella giusta, i campioni sono 3
Y = torch.tensor([2, 0, 1])

# size: nsamples x nclasses = 3x3
# queste sono le previsioni, Y_pred_good ad esempio dice che con un'alta probabilità il primo oggetto è della classe 3 (giusto), mentre Y_pred_bad dice che è della classe 1 (sbagliato)
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item(), l2.item())
print(Y.shape)
print(Y_pred_good.shape)

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1, predictions2)





# Neural network with CrossEntropyLoss (multiclass problem)
 
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linenar1(x)
        out = self.relu(out)
        out = self.linear2(out)
        #no softmax at the end!
        return out

model = NeuralNet2(input_size= 28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss() # (applies Softmax)



# Binary classification
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)  
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred

model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()
