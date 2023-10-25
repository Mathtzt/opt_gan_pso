from enum import Enum

class HeuristicNames(Enum):
    PSO = 'pso'

class DatasetNames(Enum):
    CIFAR10 = 'cifar10'

class Cifar10Classes(Enum):
    ALL = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    AIRPLANE = 'airplane'
    AUTOMOBILE = 'automobile'
    BIRD = 'bird'
    CAT = 'cat'
    DEER = 'deer'
    DOG = 'dog'
    FROG = 'frog'
    HORSE = 'horse'
    SHIP = 'ship'
    TRUCK = 'truck'

class OptimizerNames(Enum):
    SGD = 'stochastic gradient descent'
    ADAM = 'adam'