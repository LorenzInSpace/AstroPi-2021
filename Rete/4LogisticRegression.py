import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Stavolta si usa una regressione logistica, in questo caso i dati sono delle variabili cacolate da campioni di tessuto
# del seno di una paziente, lo scopo è dare la probabilità riguardante la presenza o meno di un cancro al seno.


# 0) prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

# Si divide casualmente il database dei campioni in una porzione che serve per l'allenamento, l'altra per la verifica
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 1234)

# Si usa uno StandardScaler, un oggetto che applica delle trasformazioni ai dati in modo da renderli più maneggevoi per torch
sc = StandardScaler() #gives data zero mean and unit variance
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# passa i tensori a torch
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) Definizione del modello
# deve per forza ereditare da nn.Module e usare la funzione __init()__ di quest'ultimo, poi ez
# n_input_features indica il numero di variabili in input, quindi in questo caso le caratteristiche del campione di tessuto
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module): 

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()

        # gli diamo un layer lineare
        self.linear = nn.Linear(n_input_features, 1)


    def forward(self, x):
        # applica la regressione lineare all'input, in più applica al risultato una funzione sigmoide, tipo 1/(1+e^-x), 
        # questo dà la probabilità di avere un cancro al seno
        y_predicted = torch.sigmoid(self.linear(x))
        
        return y_predicted

# crea il modello
model = LogisticRegression(n_features)


# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.BCELoss() # La loss usata qui si chiama Binary CrossEntropy, poi boh
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) # Discesa del gradiente

# 3) training loop
n_epochs = 100 # 100 ripetizioni

for epoch in range(n_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backward pass
    loss.backward()

    # updates
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

# ora si testa la rete, quindi non serve il gradiente
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
