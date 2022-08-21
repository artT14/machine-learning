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
epochs = 5