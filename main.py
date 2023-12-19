from classes.base.enums import Cifar10Classes, DatasetNames, HeuristicNames, OptimizerNames
from classes.base.namespaces import DCGANDict, DatasetDict, ExperimentDict, PSODict
from classes.experiments import Experiments

exp_dict = ExperimentDict(name = "exp",
                          description = "Utilizando PSO para otimizar topologia e hiperparâmetros da DCGAN",
                          dataset = DatasetDict(
                              type = DatasetNames.CIFAR10,
                              class_name = Cifar10Classes.AUTOMOBILE.value
                          ),
                          nexecucoes = 1,
                          heuristic_opt = PSODict(
                              name = HeuristicNames.PSO,
                            #   l_bound = [0.0009, 0, 0, 0, 0, 0, 0],
                            #   u_bound = [0.0100, 2.999, 2.999, 4.999, 3.999, 4.999, 5.999],
                              l_bound = [0, 0, 0, 0],
                              u_bound = [2.999, 4.999, 4.999, 5.999],
                              population_size = 20,
                              omega = 0.9,
                              min_speed = -3.0,
                              max_speed = 3.0,
                              cognitive_update_factor = 2.0,
                              social_update_factor = 2.0,
                              reduce_omega_linearly = True,
                              reduction_speed_factor = 1,
                              max_evaluations = 30
                          ),
                          synthesizer = DCGANDict(
                              num_epochs = 300,
                              batch_size = 8,
                              lr = 0.0005,
                              goptimizer = OptimizerNames.ADAM,
                              doptimizer = OptimizerNames.NADAM,
                              g_n_conv_blocks = 4,
                              d_n_conv_blocks = 4,
                              latent_size = 120
                          ),
                          use_ignite_temp = True)

exp = Experiments(exp_dict)
exp.run()
# exp.train_synthesize_imgs()