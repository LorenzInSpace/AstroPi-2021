# MNIST
# DataLoader, Transformation
# Multilayer Neural Net, activation function
# Loss and optimizer
# Training Loop (batch training)
# Model evaluation
# GPU support

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Si usa una convolutional neural network per poter gestire le immagini. Va usata per forza con le foto di google altrimenti ci mettiamo anni ad allenarla.
# In questo caso si hanno varie foto 32x32 a colori che devono essere classificate in base all'oggetto ritratto, stavolta non sono più 28x28.


#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

# I dati in input vanno trasformati per renderli più gestibili per torch
# dataset has PILImage images of range [0, 1]
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# se i file non ci sono li scarica, nel mentre che carica applica la trasformata transform
train_dataset = torchvision.datasets.CIFAR10(root="./Dati", train=True, download=True, transform = transform)
test_dataset = torchvision.datasets.CIFAR10(root="./Dati", train=False, download=True, transform = transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# qualche esempio
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape)
print(labels.shape)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Si crea la classe per definire la rete
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        
        self.conv1 = nn.Conv2d(3, 6, 5) # primo layer convolutional, 3 sono i canali di input, in questo caso i 3 colori R-G-B, 6 sono i canali di output, 5 definisce il kernel (un coso che serve per le convolutional, sinceramente non lo voglio toccare)
        self.pool = nn.MaxPool2d(2, 2) # presa la foto, viene fatto un campionamento e viene tipo preso il valore massimo di ogni griglia 4x4 dell'immagine, così da ridurre un pochino i dati
        self.conv2 = nn.Conv2d(6, 16, 5) # secondo layer convolutional, cosa simile al primo
        self.fc1 = nn.Linear(16*5*5, 120) #L'output di Immagine > Conv2D > Pool > Conv2D > Pool è un tensore 5x5 con 16 canali, per questo l'input è 16*5*5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # il calcolo sarebbe: immagine > Conv2D > ReLU > Pool_2x2 > Conv2D > ReLU > Pool_2x2 > Linear > ReLU > Linear > ReLU > Linear
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5) #flattens the convoluted layer, -1 vuol dire che si adatta
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# se è CUDA allora usa la scheda grafica
model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Stavolta usa SGD, non so perché non ADAM

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024 (four images per batch, three color channels, 1024 pixels)
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1) % 2000 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            

print('Finished Training')

# Test della rete neurale, come spiegato in uno dei tutorial prima
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range (10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        #max returns (value, index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

acc = 100.0 * n_correct / n_samples
print(f'Accuracy of the newtork: {acc} % \n')

for i in range(10):
    acc = 100.0 * n_class_correct[i] / n_class_samples[i]
    print(f'Accuracy of {classes[i]}: {acc} %')
