import os
import numpy as np
import torch
import torchvision
import ignite.distributed as idist

from torchvision import transforms
from torch.utils.data.dataset import Subset

from tqdm import tqdm
from optgan.classes.base.enums import Cifar10Classes

class Datasets:
        
    @staticmethod
    def get_cifar_as_dataloaders(datapath: str = "./data/", 
                                 batch_size: int = 32, 
                                 specialist_class: str = 'airplane', 
                                 nsamples: int = None,
                                 use_ignite: bool = False):
        
        train_subset, test_subset = Datasets.get_cifar_datasets(datapath, batch_size, specialist_class, nsamples)

        if use_ignite:
            train_loader = idist.auto_dataloader(
                train_subset,
                batch_size=batch_size,
                num_workers=4,
                shuffle=True,
                drop_last=True,
            )

            test_loader = idist.auto_dataloader(
                test_subset,
                batch_size=batch_size,
                num_workers=4,
                shuffle=False,
                drop_last=True,
            )
        else:
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size = batch_size, shuffle = True)
            test_loader = torch.utils.data.DataLoader(test_subset, batch_size = batch_size, shuffle = True)

        return train_loader, test_loader
    
    def get_cifar_datasets(datapath: str = "./data/",
                           batch_size: int = 32,
                           specialist_class: str = 'airplane',
                           nsamples: int = None):
        
        transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        classes = Cifar10Classes.ALL.value
        specialist_class_number = classes.index(specialist_class)

        train = torchvision.datasets.CIFAR10(root = datapath, train = True, download = True, transform = transform)
        test = torchvision.datasets.CIFAR10(root = datapath, train = False, download = True, transform = transform)

        train_indices = [i for i, (_, target) in enumerate(train) if target == specialist_class_number]
        test_indices = [i for i, (_, target) in enumerate(test) if target == specialist_class_number]
            
        train_subset = Subset(train, train_indices)
        test_subset = Subset(test, test_indices)

        if nsamples is not None:
            train_subset = Datasets.__obter_subamostra(train_subset, nsamples)
            test_subset = Datasets.__obter_subamostra(test_subset, nsamples)

        return train_subset, test_subset
    
    @staticmethod
    def get_cifar10_test_dataset(datapath: str = "./data/"):
        test_dataset = torchvision.datasets.CIFAR10(root = datapath, train = False, download = True)

        return test_dataset
        
    @staticmethod
    def __obter_subamostra(dataset, tamanho_amostra: int = 1000, seed: int = 42, nclasses: int = 10):
        np.random.seed(seed)
        # Número desejado de amostras por classe
        samples_per_class = tamanho_amostra

        # Dicionário para armazenar amostras selecionadas
        selected_samples = []

        # Encontre as amostras para cada classe
        for i in tqdm(range(nclasses)):  # 10 classes no CIFAR-10
            class_samples = [sample for sample, label in dataset if label == i]

            # Selecionar aleatoriamente o número desejado de amostras
            class_samples = torch.randperm(len(class_samples))[:samples_per_class]

            selected_samples.extend(class_samples)

        # Crie um novo Subset com as amostras selecionadas
        reduced_dataset = Subset(dataset, selected_samples)

        return reduced_dataset