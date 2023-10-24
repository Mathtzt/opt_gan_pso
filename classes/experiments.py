from math import floor
import os
import pandas as pd
from classes.base.enums import HeuristicNames, OptimizerNames

from classes.base.namespaces import ExperimentDict
from classes.helper.utils import Utils
from classes.optimizers.pso import PSO
from classes.synthesizers.dcgan.dcgan import DCGan

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
    
    def init_metaheuristic(self):
        if self.exp_dict.heuristic_opt == HeuristicNames.PSO:
            pso = PSO(omega = self.exp_dict.heuristic_opt.omega,
                      min_speed = self.exp_dict.heuristic_opt.min_speed,
                      max_speed = self.exp_dict.heuristic_opt.max_speed,
                      cognitive_update_factor = self.exp_dict.heuristic_opt.cognitive_update_factor,
                      social_update_factor = self.exp_dict.heuristic_opt.social_update_factor,
                      reduce_omega_linearly = self.exp_dict.heuristic_opt.reduce_omega_linearly,
                      reduction_speed_factor = self.exp_dict.heuristic_opt.reduction_speed_factor,
                      show_log = False)
            
            pso.main()

    def convertParams(self, params):

        learning_rate = params[0]
        goptimizer = [OptimizerNames.SGD, OptimizerNames.ADAM][floor(params[1])]
        doptimizer = [OptimizerNames.SGD, OptimizerNames.ADAM][floor(params[2])]

        return learning_rate, goptimizer, doptimizer

    def get_dcgan_gloss(self, params):
        learning_rate, goptimizer, doptimizer = self.convertParams(params)

        synthesizer = DCGan(exp_dict = self.exp_dict,
                            g_optimizer_type = goptimizer,
                            d_optimizer_type = doptimizer,
                            lr = learning_rate,
                            paths_dict = self.paths_dict)
        gloss = synthesizer.fit()

        return gloss,

    def formatParams(self, params):
        learning_rate, goptimizer, doptimizer = self.convertParams(params)
        
        return "'learning_rate'={}\n " \
               "'g_optimizer'='{}'\n " \
               "'d_optimizer'='{}'\n " \
            .format(learning_rate, goptimizer, doptimizer)