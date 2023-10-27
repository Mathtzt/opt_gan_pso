import os
import pandas as pd
import math

from classes.base.enums import HeuristicNames, OptimizerNames
from classes.base.namespaces import ExperimentDict
from classes.helper.utils import Utils
from classes.optimizers.pso import PSO
from classes.synthesizers.dcgan.dcgan import DCGan
from classes.synthesizers.dcgan.dcgan2 import DCGan as DCGAN2

class Experiments:
    def __init__(self,
            exp_dict: ExperimentDict,
            base_dir: str = "./") -> None:
        
        self.exp_dict: ExperimentDict = exp_dict
        
        self.base_dir = base_dir
        self.paths_dict = self.create_dirs()

    def create_dirs(self):
        paths_dict = {}

        paths_dict["data"] = "./data"
        paths_dict["root"] = Utils.criar_pasta(path = self.base_dir, name = "results")
        paths_dict["algo"] = Utils.criar_pasta(path = paths_dict.get("root"), name = self.exp_dict.heuristic_opt.name.value)
        paths_dict["exp"] = Utils.criar_pasta(path = paths_dict.get("algo"), name = "exp", use_date = True)

        paths_dict["model"] = Utils.criar_pasta(path = paths_dict.get("exp"), name = "model")
        paths_dict["eval_imgs"] = Utils.criar_pasta(path = paths_dict.get("exp"), name = "eval_imgs")
        paths_dict["eval_metrics"] = Utils.criar_pasta(path = paths_dict.get("exp"), name = "eval_metrics")

        return paths_dict
    
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
        goptimizer = [OptimizerNames.SGD, OptimizerNames.ADAM][math.floor(params[1])]
        doptimizer = [OptimizerNames.SGD, OptimizerNames.ADAM][math.floor(params[2])]

        return learning_rate, goptimizer, doptimizer

    def get_dcgan_gloss(self, params):
        learning_rate, goptimizer, doptimizer = self.convert_params(params)

        synthesizer = DCGan(exp_dict = self.exp_dict,
                            g_optimizer_type = goptimizer,
                            d_optimizer_type = doptimizer,
                            lr = learning_rate,
                            paths_dict = self.paths_dict)
        gloss = synthesizer.fit()

        return gloss,

    def format_params(self, params):
        learning_rate, goptimizer, doptimizer = self.convert_params(params)
        
        return "'learning_rate'={}\n " \
               "'g_optimizer'='{}'\n " \
               "'d_optimizer'='{}'\n " \
            .format(learning_rate, goptimizer, doptimizer)
    
    def train_synthesize_imgs(self):
        # synthesizer = DCGan(exp_dict = self.exp_dict,
        #                     g_optimizer_type = self.exp_dict.synthesizer.goptimizer,
        #                     d_optimizer_type = self.exp_dict.synthesizer.doptimizer,
        #                     lr = self.exp_dict.synthesizer.lr,
        #                     paths_dict = self.paths_dict)
        
        synthesizer = DCGAN2(exp_dict = self.exp_dict,
                             g_optimizer_type = self.exp_dict.synthesizer.goptimizer,
                             d_optimizer_type = self.exp_dict.synthesizer.doptimizer,
                             lr = self.exp_dict.synthesizer.lr,
                             paths_dict = self.paths_dict)
        
        synthesizer.fit()
        synthesizer.generate_synthetic_images(exp_dirname = str(self.paths_dict.get("exp")).split("/")[3])