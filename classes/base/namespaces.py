from typing import NamedTuple

from classes.base.enums import DatasetsNames, HeuristicNames, OptimizerNames

class DatasetDict(NamedTuple):
    type: DatasetsNames
    class_number: int

class PSODict(NamedTuple):
    name: HeuristicNames
    l_bound: list
    u_bound: list
    # dimensions: int
    population_size: int
    # bounds: list[int, int]
    omega: float
    min_speed: float
    max_speed: float
    cognitive_update_factor: float
    social_update_factor: float
    reduce_omega_linearly: bool
    reduction_speed_factor: float

class ExperimentDict(NamedTuple):
    name: str
    dataset: DatasetDict
    nexecucoes: int
    heuristic_opt: PSODict
    goptimizer: OptimizerNames
    doptimizer: OptimizerNames
