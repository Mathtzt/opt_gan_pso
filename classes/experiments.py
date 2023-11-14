import os
import pandas as pd
import math

from classes.base.enums import DatasetNames, HeuristicNames, OptimizerNames
from classes.base.namespaces import ExperimentDict
from classes.helper.datasets import Datasets
from classes.helper.vis import Vis
from classes.helper.utils import Utils
from classes.optimizers.pso import PSO
from classes.synthesizers.dcgan.dcgan import DCGan
from classes.synthesizers.dcgan.dcgan2 import DCGan as DCGAN2
from classes.synthesizers.dcgan.ig_dcgan import DCGanIgnite
from tqdm import tqdm

class Experiments:
    def __init__(self,
            exp_dict: ExperimentDict,
            base_dir: str = "./") -> None:
        
        self.exp_dict: ExperimentDict = exp_dict
        
        self.base_dir = base_dir
        self.paths_dict: dict = self.create_dirs()
        self.data: tuple = self.load_data()
        
        self.generate_test_data_imgs()

    def create_dirs(self):
        paths_dict = {}

        paths_dict["data"] = "./data"
        paths_dict["root"] = Utils.criar_pasta(path = self.base_dir, name = "results")
        paths_dict["algo"] = Utils.criar_pasta(path = paths_dict.get("root"), name = self.exp_dict.heuristic_opt.name.value)
        paths_dict["exp"] = Utils.criar_pasta(path = paths_dict.get("algo"), name = "exp", use_date = True)

        paths_dict["model"] = Utils.criar_pasta(path = paths_dict.get("exp"), name = "model")
        paths_dict["eval_imgs"] = Utils.criar_pasta(path = paths_dict.get("exp"), name = "eval_imgs")
        paths_dict["eval_metrics"] = Utils.criar_pasta(path = paths_dict.get("exp"), name = "eval_metrics")

        paths_dict["fake_imgs"] = Utils.criar_pasta(path = paths_dict.get("eval_imgs"), name = f"{self.exp_dict.dataset.class_name}_imgs")

        return paths_dict
    
    def load_data(self):
        print("Carregando dados...")
        if self.exp_dict.dataset.type == DatasetNames.CIFAR10:

            train_loader, test_loader = Datasets.get_cifar_as_dataloaders(datapath = self.paths_dict["data"],
                                                                          batch_size = self.exp_dict.synthesizer.batch_size,
                                                                          specialist_class = self.exp_dict.dataset.class_name,
                                                                          nsamples = None,
                                                                          use_ignite = self.exp_dict.use_ignite_temp)
            return (train_loader, test_loader)                
        else:
            raise Exception("data_type não implementado ainda.")
    
    def generate_test_data_imgs(self):
        Vis.gen_and_save_img(self.paths_dict["data"])

    def run(self):

        if self.exp_dict.heuristic_opt.name == HeuristicNames.PSO:
            pso = PSO(exp_dict = self.exp_dict,
                      paths_dict = self.paths_dict,
                      show_log = False)
            
            pso.main(func_ = self.get_dcgan_gloss)
            print("######## RESULTADO ########\n")
            print(self.format_params(pso.best))

    def convert_params(self, params):
        learning_rate = params[0]
        goptimizer = [OptimizerNames.SGD, OptimizerNames.ADAM, OptimizerNames.NADAM][math.floor(params[1])]
        doptimizer = [OptimizerNames.SGD, OptimizerNames.ADAM, OptimizerNames.NADAM][math.floor(params[2])]
        batch_size = [8, 16, 32, 64, 128][math.floor(params[3])]
        latent_size = [10, 20, 50, 100][math.floor(params[4])]

        return learning_rate, goptimizer, doptimizer, batch_size, latent_size

    def get_dcgan_gloss(self, params):
        learning_rate, goptimizer, doptimizer, batch_size, latent_size = self.convert_params(params)

        if self.exp_dict.use_ignite_temp:
            synthesizer = DCGanIgnite(train_loader = self.data[0],
                                      test_loader = self.data[1],
                                      exp_dict = self.exp_dict,
                                      paths_dict = self.paths_dict,
                                      g_optimizer_type = self.exp_dict.synthesizer.goptimizer,
                                      d_optimizer_type = self.exp_dict.synthesizer.doptimizer,
                                      lr = self.exp_dict.synthesizer.lr)
            
            obj_val = synthesizer.fit(save_files = False)

        else:
            synthesizer = DCGAN2(exp_dict = self.exp_dict,
                                paths_dict = self.paths_dict,
                                g_optimizer_type = goptimizer,
                                d_optimizer_type = doptimizer,
                                lr = learning_rate,
                                batch_size = batch_size,
                                latent_size = latent_size)
        
            obj_val = synthesizer.fit(train_loader = self.data[0],
                                      test_loader = self.data[1],
                                      save_files = False)

        return obj_val,

    def format_params(self, params):
        learning_rate, goptimizer, doptimizer, batch_size, latent_size = self.convert_params(params)
        
        return "'learning_rate'={}\n " \
               "'g_optimizer'='{}'\n " \
               "'d_optimizer'='{}'\n " \
               "'batch_size'='{}'\n " \
               "'latent_size'='{}'\n " \
            .format(learning_rate, goptimizer, doptimizer, batch_size, latent_size)
    
    def train_synthesize_imgs(self):
        # synthesizer = DCGan(exp_dict = self.exp_dict,
        #                     g_optimizer_type = self.exp_dict.synthesizer.goptimizer,
        #                     d_optimizer_type = self.exp_dict.synthesizer.doptimizer,
        #                     lr = self.exp_dict.synthesizer.lr,
        #                     paths_dict = self.paths_dict)
        
        if self.exp_dict.use_ignite_temp:
            synthesizer = DCGanIgnite(train_loader = self.data[0],
                                      test_loader = self.data[1],
                                      exp_dict = self.exp_dict,
                                      paths_dict = self.paths_dict,
                                      g_optimizer_type = self.exp_dict.synthesizer.goptimizer,
                                      d_optimizer_type = self.exp_dict.synthesizer.doptimizer,
                                      lr = self.exp_dict.synthesizer.lr)
            synthesizer.fit()

        else:
            synthesizer = DCGAN2(exp_dict = self.exp_dict,
                                paths_dict = self.paths_dict,
                                g_optimizer_type = self.exp_dict.synthesizer.goptimizer,
                                d_optimizer_type = self.exp_dict.synthesizer.doptimizer,
                                lr = self.exp_dict.synthesizer.lr)
        
            synthesizer.fit()
            synthesizer.generate_synthetic_images(exp_dirname = str(self.paths_dict.get("exp")).split("/")[3])