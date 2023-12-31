from typing import NamedTuple

from classes.base.enums import DatasetNames, HeuristicNames, OptimizerNames

class DatasetDict(NamedTuple):
    type: DatasetNames
    class_name: str

class PSODict(NamedTuple):
    name: HeuristicNames
    l_bound: list
    u_bound: list
    population_size: int
    omega: float
    min_speed: float
    max_speed: float
    cognitive_update_factor: float
    social_update_factor: float
    reduce_omega_linearly: bool
    reduction_speed_factor: float
    max_evaluations: int

class DCGANDict(NamedTuple):
    num_epochs: int
    batch_size: int
    lr: float
    goptimizer: OptimizerNames
    doptimizer: OptimizerNames
    g_n_conv_blocks: int
    d_n_conv_blocks: int
    latent_size: int

class ExperimentDict(NamedTuple):
    name: str
    description: str
    dataset: DatasetDict
    nexecucoes: int
    heuristic_opt: PSODict
    synthesizer: DCGANDict
    use_ignite_temp: bool