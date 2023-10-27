import os
import numpy as np

import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Subset

from classes.base.enums import Cifar10Classes

class Datasets:
        
    @staticmethod
    def get_cifar_as_dataloaders(datapath: str = "./data/", batch_size: int = 32, specialist_class: str = 'airplane'):
        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        classes = Cifar10Classes.ALL.value
        specialist_class_number = classes.index(specialist_class)

        train = torchvision.datasets.CIFAR10(root = datapath, train = True, download = True, transform = transform)
                
        indices = [i for i, (_, target) in enumerate(train) if target == specialist_class_number]
        subset = Subset(train, indices)

        dataloader = torch.utils.data.DataLoader(subset, batch_size = batch_size, shuffle = True)

        return dataloader