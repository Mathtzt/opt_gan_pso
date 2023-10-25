from classes.base.enums import Cifar10Classes, DatasetNames, HeuristicNames
from classes.base.namespaces import DatasetDict, ExperimentDict, PSODict
from classes.experiments import Experiments

exp_dict = ExperimentDict(name = "exp",
                          description = "Utilizando PSO para otimizar topologia e hiperparâmetros da DCGAN",
                          dataset = DatasetDict(
                              type = DatasetNames.CIFAR10,
                              class_name = Cifar10Classes.AIRPLANE.value
                          ),
                          nexecucoes = 1,
                          heuristic_opt = PSODict(
                              name = HeuristicNames.PSO,
                              l_bound = [0.0004, 0, 0],
                              u_bound = [0.01, 1.999, 1.999],
                              population_size = 10,
                              omega = 0.9,
                              min_speed = -3.0,
                              max_speed = 3.0,
                              cognitive_update_factor = 2.0,
                              social_update_factor = 2.0,
                              reduce_omega_linearly = True,
                              reduction_speed_factor = 1
                          ))

exp = Experiments(exp_dict)
exp.run()