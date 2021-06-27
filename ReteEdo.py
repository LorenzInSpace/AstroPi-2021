import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import os
import pandas as pd
from skimage import io
from torch.utils.data import (
    Dataset,
    DataLoader,
)
import matplotlib.pyplot as plt
import sys

# per farlo funzionare prima dovete installare tensorboard, scrivere poi sul prompt: "tensorboard --logdir=runs", successivamente dovrebbe dirvi che ha creato un localhost a una certa porta, la mia è 6006. Accedete col browser e dovrebbe mostrarvi
# tensorboard. A quel punto fate partire il programma
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/Project5")

# Define the dataset class
class Astropi(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

dataset = Astropi(
    # Our database is stored in a folder called "DatasetAstropi" on the same level as the .py code:
    # The folder includes a csv file called "label_data.csv" and another folder ("images") containing
    # the images to be submitted to the neural network. Each image is named "image_NUM.jpg"
    csv_file="DatasetAstropi/label_data.csv",
    root_dir="DatasetAstropi/images",
    transform=transforms.ToTensor(),
)

# Allow for GPU usage when available. We didn't end up using it in our analisys.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters for the NN
num_epochs = 10
batch_size = 5
learning_rate = 0.0005

# Load the dataset and split it between a train and a test set
train_set, test_set = torch.utils.data.random_split(dataset, [568, 150])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

"""
examples = iter(test_loader)
example_data, example_targets = examples.next()

prova = example_data.permute(1,0,2,3)
prova = prova[0]
print(prova.shape)
print(example_data.shape)
# for i in range(6):
#     plt.subplot(2,3, i+1)
#     image = example_data[i].permute(1,2,0)
#     plt.imshow(image)
#plt.show()
img_grid = torchvision.utils.make_grid(example_data)
print(img_grid.shape)
writer.add_image('ISS_images',img_grid)
#writer.close()
#sys.exit()
"""

# Defines the network structure
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Define the layers and functions of the network:
        # pool allows us to reduce the image size by only considering the maximum valued pixel in an area
        # a convolutional layer passes information of a pixel by considering the neighboring ones
        # a linear layer outputs a linear combination of the input layers to each output layer
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(3, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 20, 5)
        self.conv4 = nn.Conv2d(20, 33, 5)
        self.fc1 = nn.Linear(65208,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # Describes the order of operations, relu is used to render this a Nonlinear system
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(torch.sigmoid(self.conv2(x)))
        x = self.pool2(torch.sigmoid(self.conv3(x)))
        x = self.pool2(F.relu(self.conv4(x)))
        x = x.view(-1, 65208)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = ConvNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Stavolta usa SGD, non so perché non ADAM
criterion = nn.MSELoss()

def evaluates():
    # Evaluates the network on the test set. We decided to output the loss because
    # it can be easily calculated when comparing floating point values
    with torch.no_grad():
        cont = 0
        loss = 0
        for i, (images, labels) in enumerate(test_loader):
            loss += criterion(model(images), labels.to(torch.float32).view(-1, 1))
            cont +=1
        print(f'Mean Squared Error {loss/cont:.8f}')

#writer.add_graph(model, example_data)
#writer.close()
#sys.exit()
n_total_steps = len(train_loader)

running_loss = 0.0
print("funziona")
for epoch in range(num_epochs):
    evaluates()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        loss = criterion(model(images), labels.to(torch.float32).view(-1,1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the progress of training
        running_loss = loss.item()
        if(i+1) % 10 == 0:
             print(f'Epoch {epoch+1}/{num_epochs}, Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
             writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
             running_loss = 0
print('Finished Training')
evaluates()

# Save the trained model for future use
torch.save(model.state_dict(), "model")
