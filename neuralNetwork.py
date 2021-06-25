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
batch_size = 1
learning_rate = 0.001

# Load the dataset and split it between a train and a test set
train_set, test_set = torch.utils.data.random_split(dataset, [700, 18])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Defines the network structure
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Define the layers and functions of the network:
        # pool allows us to reduce the image size by only considering the maximum valued pixel in an area
        # a convolutional layer passes information of a pixel by considering the neighboring ones
        # a linear layer outputs a linear combination of the input layers to each output layer
        self.pool1 = nn.MaxPool2d(3, 3) 
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 20, 5)
        self.conv4 = nn.Conv2d(20, 33, 5)
        self.fc1 = nn.Linear(65208,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # Describes the order of operations, relu is used to render this a Nonlinear system
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool2(F.relu(self.conv4(x)))
        x = x.view(-1, 65208)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        loss = ((model(images) - labels)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print the progress of training
        if(i+1) % int(0.25*len(train_loader)) == 0:
             print(f'Epoch {epoch+1}/{num_epochs}, Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print('Finished Training')

# Save the trained model for future use
torch.save(model.state_dict(), "model")

# Evaluates the network on the test set. We decided to output the loss because 
# it can be easily calculated when comparing floating point values
with torch.no_grad():
    cont = 0
    loss = 0
    for i, (images, labels) in enumerate(test_loader):
        loss += ((model(images) - labels)**2).mean().item()
        cont +=1
    print(f'Mean Squared Error {loss/cont:.8f}')
