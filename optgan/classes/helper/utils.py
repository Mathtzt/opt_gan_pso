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
        return transforms.Normalize(mean= [-m/s for m, s in zip(mean, std)],
                                    std= [1/s for s in std])
    
    @staticmethod
    def vector_to_images(vectors, nchannel: int = 3, dsize: tuple = (32, 32)):
        return vectors.view(vectors.size(0), nchannel, dsize[0], dsize[1])
    
    @staticmethod
    def obter_log_filename(nome_base):
        dt = datetime.now()
        day = dt.strftime("%d")
        mes = dt.strftime("%m")
        hour = dt.strftime("%H")
        mm = dt.strftime("%M")
        
        dirname_base = f"_{day}{mes}_{hour}{mm}.txt"

        return nome_base + dirname_base

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
                fp.write("%s\n" % str(item))

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
        
    @staticmethod
    def save_experiment_as_csv(base_dir: str, dataframe: pd.DataFrame, filename: str):
        BASE_DIR = base_dir
        FILE_PATH = BASE_DIR + '/' + filename + '.csv'
        if not os.path.exists(FILE_PATH):
            dataframe.to_csv(FILE_PATH, index = False)
        else:
            df_loaded = pd.read_csv(FILE_PATH)
            dataframe_updated = pd.concat([df_loaded, dataframe], axis = 0)

            dataframe_updated.to_csv(FILE_PATH, index = False)

    @staticmethod
    def rgb_to_gray(image):
        # Converte a imagem para escala de cinza usando média ponderada dos canais RGB
        grayscale_image = torch.mean(image, dim=0).unsqueeze(0)

        return grayscale_image
    
    @staticmethod
    # Função para calcular a PDF de uma classe específica em escala de cinza
    def calculate_pdf(dataset, class_label):
        # Filtra o dataset para conter apenas amostras da classe escolhida
        class_dataset = [sample for sample in dataset if sample[1] == class_label]

        # Obtenha os valores de pixel
        pixel_values = [sample[0].numpy().flatten() for sample in class_dataset]

        # Concatene os valores de pixel
        pixel_values = np.concatenate(pixel_values)

        # Calcule a PDF das intensidades de pixel
        hist, bins = np.histogram(pixel_values, bins=256, range=(0, 1), density=True)
        pdf = hist / np.sum(hist)

        return pdf, bins, class_label