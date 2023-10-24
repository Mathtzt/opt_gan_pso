import os
import pandas as pd 
import numpy as np
import json
import torch

from torchvision import transforms
from datetime import datetime
from tqdm import tqdm

class Utils:
    
    @staticmethod
    def get_device_available():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod 
    def inverse_normalize(mean = (0.5,), std = (0.5,)):
        #TODO CIFAR10 mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)
        return transforms.Normalize(mean= [-m/s for m, s in zip(mean, std)],
                                    std= [1/s for s in std])

    @staticmethod
    def criar_pasta(path, name, use_date = False):
        """
        Método responsável por criar a pasta no diretório passado como parâmetro.
        """
        if use_date:
            dt = datetime.now()
            day = dt.strftime("%d")
            mes = dt.strftime("%m")
            hour = dt.strftime("%H")
            mm = dt.strftime("%M")
            dirname_base = f"_{day}{mes}_{hour}{mm}"
            directory = name + dirname_base
        else:
            directory = name

        parent_dir = path

        full_path = os.path.join(parent_dir, directory)

        if os.path.isdir(full_path):
            return full_path
        else:
            os.mkdir(full_path)
            return full_path
        
    @staticmethod    
    def save_list_as_txt(l: list, path: str = './', name: str = 'loss', log: bool = True):
        with open(f'{path}/{name}.txt', 'w') as fp:
            for item in l:
                fp.write("%s\n" % item)

        if log:
            print(f'File {name} was saved.')

    @staticmethod
    def open_list_from_txt(path: str = './', name: str = 'loss', values_type = 'float'):
        loaded_list = []

        with open(f'{path}/{name}.txt', 'r') as fp:
            for line in fp:
                x = line[:-1]

                if values_type == 'float':
                    loaded_list.append(float(x))
                elif values_type == 'int':
                    loaded_list.append(int(x))
                elif values_type == 'list':
                    loaded_list.append(eval(x))
                else:
                    loaded_list.append(x)
        return loaded_list
    
    @staticmethod
    def save_dict_as_txt(path: str, dic: dict, name: str):
        with open(f'{path}/{name}_metrics.txt', 'w') as f:
            json.dump(dic, f)

    @staticmethod
    def load_dict_from_txt(path: str, name: str):
        with open(f'{path}/{name}_metrics.txt', 'r') as f:
            dic = json.load(f)
            
            return dic