# LINK: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
# STEPS OF TRAINING A MODEL:
#   in each ITERATION/EPOCH:
#   - make guess about output
#   - calculate the error in guess (loss)
#   - collect derivatives of the err w/ respect to its parameters
#   - optimize these parameters using gradient descent

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# PREP DATASET
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# SET UP DATALOADER
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# DEFINE & SET UP MODEL
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# HYPER PARAMETERS - control the optimization process
learning_rate = 1e-3
batch_size = 64 # num of data samples propogated thru NN before parameters are updated
epochs = 10

# OPTIMIZATION LOOP
#   - Train Loop: iterate over training dataset & try to converge to optimal parameters
#   - Test Loop: iterate over test dataset to check if model performance is improving

# LOSS FUNCTION: goal is to minimize error in this function
loss_fn = nn.CrossEntropyLoss()

# OPTIMIZER: process of adjusting model parameters to reduce model error in each training step
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# FULL IMPLEMENTATION
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred,y)
        # Backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")