import os
import pickle
import pandas as pd

from abc import ABC, abstractmethod

class Base(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self):
        pass
    
    def save(self, model, filename: str, path_dir: str):
        full_path = os.path.join(path_dir, filename)

        pickle.dump(model, open(f"{full_path}.pkl", "wb"))

    def load(self, filename: str, path_dir: str):
        full_path = os.path.join(path_dir, filename)

        return pickle.load(open(f"{full_path}.pkl", "rb"))