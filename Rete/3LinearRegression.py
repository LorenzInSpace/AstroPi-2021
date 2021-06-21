# si usano i tensori e i metodi di torch per fare una regressione lineare a modo

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) vengono creati dei tensori che simulano una regressione lineare, hanno 100 elementi
X_numpy, y_numpy = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 1)

# Si creano dei tensori di torch a partire da quelli di numpy, così da poterci lavorare
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

# Numero di campioni e variabili
n_samples, n_features = X.shape


# 1) model
input_size = n_features
output_size = 1
#crea un modello di regressione lineare (in pratica una funzione lineare)
model = nn.Linear(input_size, output_size)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss() # Usa il Mean Squared Error
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) # Al posto di aggiornare il gradiente manualmente si usa l'ottimizzatore, che fa tutto in automatico

# 3) Allenamento
n_epochs = 100

for epoch in range(n_epochs):
    #forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    #backward pass
    loss.backward()

    #update
    optimizer.step()

    #reset gradient
    optimizer.zero_grad()


    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')


# plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, "ro")
plt.plot(X_numpy, predicted, "b")
plt.show()
