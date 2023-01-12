import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as trans
from torchvision import datasets as dtsets

traindt = dtsets.MNIST(root='./data', 
                            train=True, 
                            transform=trans.ToTensor(),
                            download=True)

testdt = dtsets.MNIST(root='./data', 
                           train=False, 
                           transform=trans.ToTensor(),download=True)
                