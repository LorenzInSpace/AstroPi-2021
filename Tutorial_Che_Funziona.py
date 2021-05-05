import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1296*972, 512), # strati, dati input - output
            
            #TODO CONVOLUTIONAL
            
            nn.ReLU(), # funzione sigmoide, rectified linear unit
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x) # trasforma matrice in un array
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device) # device è o cpu o gpu
print(model)


loss_fn = nn.CrossEntropyLoss() # per capire quanto sbaglia rispetto alla risposta giusta.
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # Implements stochastic gradient descent
                                                #learning rate


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y) # differenza rispetto a quello che doveva essere

        # Backpropagation
        optimizer.zero_grad() # resetta
        loss.backward() # calcola la derivata del costo rispetto alle x
        optimizer.step() # fa step

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    #test_loss, correct = 0, 0
    with torch.no_grad(): # azzera il gradiente del blocco di codice, riduce i consumi di memoria
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            
            # prendi i nodo piu acceso, se è uguale a y allora aggiungi 1 al numero di corretti.
            #print("pred arg max:",pred.argmax(1))
            #print("y:",y )
            #print("DEBUG:",pred.argmax(1) == y)
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    #correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")

#torch.save(model.state_dict(), "model.pth")
#print("Saved PyTorch Model State to model.pth")



"""classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
"""
