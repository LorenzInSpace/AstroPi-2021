import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
import torch.nn.functional as F
import os
import pandas as pd
from skimage import io
from torch.utils.data import (
    Dataset,
    DataLoader,
)


class Astropi(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path) # LA MALEDETTA
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

dataset = Astropi(
    # Serve una cartella DatasetAstropi nella stessa directory del codice
    # dentro la cartella mettete il csv che ho mandato e tutte le immagini in una cartella images
    csv_file="DatasetAstropi/label_data.csv",
    root_dir="DatasetAstropi/images",
    transform=transforms.ToTensor(),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cambiate come torna meglio
num_epochs = 10 # passi di ottimizzazione
batch_size = 1 # quante immagini processate allo stesso momento
learning_rate = 0.001

# Carica le immagini, cambiate lo split come torna meglio
train_set, test_set = torch.utils.data.random_split(dataset, [700, 18])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Si crea la classe per definire la rete
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # La prima volta che fate girare il codice viene errore,
        # assecondate quello che vi dice deve essere la dimensione
        # giusta al posto di 16*5*5, da lì cambiate come volete
        self.conv1 = nn.Conv2d(3, 6, 5) # primo layer convolutional, 3 sono i canali di input, in questo caso i 3 colori R-G-B, 6 sono i canali di output, 5 definisce il kernel (un coso che serve per le convolutional, sinceramente non lo voglio toccare)
        self.pool1 = nn.MaxPool2d(3, 3) # presa la foto, viene fatto un campionamento e viene tipo preso il valore massimo di ogni griglia 4x4 dell'immagine, così da ridurre un pochino i dati
        self.pool2 = nn.MaxPool2d(2, 2) # presa la foto, viene fatto un campionamento e viene tipo preso il valore massimo di ogni griglia 4x4 dell'immagine, così da ridurre un pochino i dati
        self.conv2 = nn.Conv2d(6, 16, 5) # secondo layer convolutional, cosa simile al primo
        self.conv3 = nn.Conv2d(16, 20, 5)
        self.conv4 = nn.Conv2d(20, 33, 5)
        self.fc1 = nn.Linear(65208,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool2(F.relu(self.conv4(x)))

        # Mettete lo stesso numero di prima
        x = x.view(-1, 65208)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x

# se è CUDA allora usa la scheda grafica
model = ConvNet().to(device)

# Se capite come fargli usare le funzioni di errore loro tanto meglio
#criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Stavolta usa SGD, non so perché non ADAM

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #loss = criterion(model(images) , labels) # non so perché ma non mi funziona lol
        loss = ((model(images) - labels)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1) % int(0.25*len(train_loader)) == 0:
             print(f'Epoch {epoch+1}/{num_epochs}, Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print('Finished Training')

torch.save(model.state_dict(), "model")

with torch.no_grad():
    cont = 0
    loss = 0
    for i, (images, labels) in enumerate(test_loader):
        #loss += criterion(model(images), labels).item() # non mi funziona lol
        loss += ((model(images) - labels)**2).mean().item()
        cont +=1
    print(f'Mean Squared Error {loss/cont:.8f}')
