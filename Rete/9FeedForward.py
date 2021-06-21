# MNIST
# DataLoader, Transformation
# Multilayer Neural Net, activation function
# Loss and optimizer
# Training Loop (batch training)
# Model evaluation
# GPU support (Io uso la CPU, ho paura mi frigga la scheda grafica, basta installare la versione normale di torch)

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# viste le cose imparate fin ora, si usa il dataset MNIST. Sono foto 28x28 di numeri da 0 a 9 (a volte scritti anche male) e la rete ha il compito di riconoscere quale numero si tratta


#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # da me è un casino


# hyper parameters
input_size = 784 # 28x28, le dimensioni delle foto. Quindi ad ogni pixel è associato un neurone del layer di input
hidden_size = 100
num_classes = 10 # i numeri da 0 a 9
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = torchvision.datasets.MNIST(root ="./Dati", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root ="./Dati", train=False, transform=transforms.ToTensor())
# Shape = (100, 1, 28, 28)

# Per gestire i dati che vengono dati in input alla rete si usano i DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#---piccolo esempio---
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape)
print(labels.shape)

for i in range(6):
    plt.subplot(2,3, i+1)
    plt.imshow(samples[i][0], cmap="gray")
plt.show()


# Si definisce la rete
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out
        # Se usiamo come loss CrossEntropy allora quest'ultima calcola in automatico softmax, così non dobbiamo applicarlo qui

model = NeuralNet(input_size, hidden_size, num_classes)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Non si usa SGD (Stochastic Gradient Descent) ma Adam, si vede funziona meglio


#training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  # da train_loader prende un gruppetto dato da batch_size, viene diviso in images (input) e lables (l'output desiderato)
      
        # (100, 1, 28, 28) reshaped into (100, 784). Quell'1 viene dal fatto che le immagini sono in bianco e nero, quindi c'è un solo canale, se fossero state a colori avremmo avuto un 3
        images = images.reshape(-1, 28*28).to(device) # il -1 perché così torch capisce da sé le dimensioni da dare affinché torni con il 28*28
        labels = labels.to(device)

        #forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')



#test, non serve il gradiente
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1) # outputs contiene le probabilità che la foto in input sia uno dei vari numeri (se tipo la foto fosse di un 1 e la rete fosse allenata bene l'output sarebbe tipo [0.9, 0.1, 0.05, 0.2, ecc.]. Max estrae prima il valore massimo, e poi l'indice del valore massimo, a noi interessa solo il secondo e quindi buttiamo via il primo
        n_samples += labels.shape[0] # si contano i campioni
        n_correct += (predictions == labels).sum().item() # si confrontano i tensori dei numeri previsti e di quelli effettivi e si contano quanti sono azzeccati, se tipo avessimo [1,2,3,4,5] e [1,0,2,4,5], il tensore risultante sarebbe [1,0,0,1,1], quindi 3 giusti da aggiungere al totale
    
    acc = 100.0 * n_correct / n_samples # si guarda la percentuale giusti/totali
    print(f'accuracy = {acc}')

