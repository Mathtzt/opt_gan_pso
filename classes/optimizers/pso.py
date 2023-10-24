import pandas as pd
import numpy as np
from classes.base.algoritmo import Algoritmo

from deap import base, creator, tools

class PSO:
    def __init__(self,
                #  dimensions: int = 10,
                #  population_size: int = 10,
                #  bounds: list = [-100, 100],
                 omega: float = .9,
                 min_speed: float = -0.5,
                 max_speed: float = 3.,
                 cognitive_update_factor: float = 2.,
                 social_update_factor: float = 2.,
                 reduce_omega_linearly: bool = False,
                 reduction_speed_factor: float = 1,
                 show_log: bool = True):

        self.omega = omega
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.cognitive_update_factor = cognitive_update_factor
        self.social_update_factor = social_update_factor
        self.reduce_omega_linearly = reduce_omega_linearly
        self.reduction_speed_factor = reduction_speed_factor
        self.show_log = show_log

        self.nout_bounds = 0
        self.toolbox = base.Toolbox()

    def main(self, 
             func_name: ProblemFuncNames,
             nexecucao: int = 0, 
             exp_path: str = './',
             imgs_path: str = './'):
        func_ = self.obter_func_objetivo(func_name = func_name)
        self.toolbox.register(alias = 'evaluate', 
                              function = func_[0].evaluate)
        
        ## inicializações
        self.define_as_minimization_problem()
        self.creating_particle_class()
        self.create_particle()
        self.creating_particle_register()
        self.register_to_update_particles()

        ## criando a população
        population = self.toolbox.populationCreator(n = self.population_size)

        ## criando objeto para salvar as estatísticas
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('min', np.min)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'evals'] + stats.fields

        ## loop
        best = None
        omega = self.omega
        igeneration_stopped = 0
        nevaluations = 0
        finish_optimization = False

        best_fitness_history = []
        avg_fitness_history = []
        avg_euclidian_distance_history = []

        for idx, generation in enumerate(range(1, self.max_evaluations + 1)):
            # reduzindo omega linearmente
            if self.reduce_omega_linearly:
                omega = self.omega - (idx * (self.omega - 0.4) / (self.max_evaluations * self.reduction_speed_factor))
            
            # avaliar todas as partículas na população
            for particle in population:
                # calcular o valor de fitness da partícula / avaliação
                particle.fitness.values = (self.toolbox.evaluate(particle), )
                # atualizando melhor partícula global
                if particle.best is None or particle.best.size == 0 or particle.best.fitness < particle.fitness:
                    particle.best = creator.Particle(particle)
                    particle.best.fitness.values = particle.fitness.values
                # atualizando valor global
                if best is None or best.size == 0 or best.fitness < particle.fitness:
                    best = creator.Particle(particle)
                    best.fitness.values = particle.fitness.values
                # atualizando número de avaliações, verificando limites de tempo de otimização permitidos
                nevaluations += 1
                stop_cond1 = abs(best.fitness.values[0] - func_[1]) < 10e-8
                stop_cond2 = nevaluations >= self.max_evaluations
                if stop_cond1 or stop_cond2:
                    finish_optimization = True
                    igeneration_stopped = idx
                    break

            # atualizando velocidade e posição
            for particle in population:
                self.toolbox.update(particle, best, omega)

            avg_euclidian_distance_history.append(self.calc_distancia_euclidiana_media_da_populacao(population))
            # salvando as estatísticas
            if generation == 1 or generation % 500 == 0:
                best_fitness_history.append(best.fitness.values)
                logbook.record(gen = generation,
                               evals = len(population),
                               **stats.compile(population))
                if self.show_log:
                    if self.reduce_omega_linearly:
                        print(logbook.stream + f" | omega = {omega}")
                    else:
                        print(logbook.stream)
            
            if finish_optimization:
                break

        avg_fitness_history = [logbook[i]['avg'] for i in range(len(logbook))]

        self.print_informacoes_gerais_optimizacao(best, igeneration_stopped)
        self.criar_grafico_evolucao_fitness(hist_best_fitness = best_fitness_history,
                                            hist_avg_fitness = avg_fitness_history,
                                            imgs_path = imgs_path, 
                                            img_name = f"pso_exec_{nexecucao}",
                                            use_log = True)
        self.criar_grafico_evolucao_distancia_media_pontos(hist_dist_pontos = avg_euclidian_distance_history,
                                                           imgs_path = imgs_path,
                                                           img_name = f"pso_distance_particles_{nexecucao}")
        self.criar_registro_geral(nexecucao = nexecucao,
                                  func_objetivo = func_name.value,
                                  best_particle = best,
                                  best_fitness = best.fitness.values[0],
                                  igeneration_stopped = igeneration_stopped,
                                  exp_path = exp_path)
        
        self.salvar_historico_em_arquivo_txt(best_fitness_history, exp_path, f'best_fitness_{nexecucao}')
        self.salvar_historico_em_arquivo_txt(avg_fitness_history, exp_path, f'avg_fitness_{nexecucao}')
        self.salvar_historico_em_arquivo_txt(avg_euclidian_distance_history, exp_path, f'dist_particles_{nexecucao}')

        del creator.FitnessMin
        del creator.Particle

    def print_informacoes_gerais_optimizacao(self, best, igeneration_stopped):
        print("-- Melhor partícula = ", best)
        print("-- Melhor fitness = ", best.fitness.values[0])
        print("-- Geração que parou a otimização = ", igeneration_stopped)
        print("-- Qtd de vezes que saiu do espaço de busca = ", self.nout_bounds)

    def define_as_minimization_problem(self):
        creator.create(name = "FitnessMin",
                       base = base.Fitness,
                       weights = (-1., ))
        
    def creating_particle_class(self):
        creator.create(name = 'Particle',
                       base = np.ndarray,
                       fitness = creator.FitnessMin,
                       speed = None,
                       best = None)
        
    def create_particle(self):
        particle = creator.Particle(np.random.uniform(low = self.bounds[0],
                                                      high = self.bounds[1],
                                                      size = self.dimensions))
        
        particle.speed = np.random.uniform(low = self.min_speed,
                                           high = self.max_speed,
                                           size = self.dimensions)
        
        return particle

    def creating_particle_register(self):
        self.toolbox.register(alias = 'particleCreator',
                              function = self.create_particle)
                        
        self.toolbox.register('populationCreator', tools.initRepeat, list, self.toolbox.particleCreator)

    def update_particle(self, particle, best, omega):
        local_update_factor = self.cognitive_update_factor * np.random.uniform(0, 1, particle.size)
        global_update_factor = self.social_update_factor * np.random.uniform(0, 1, particle.size)

        local_speed_update = local_update_factor * (particle.best - particle)
        global_speed_update = global_update_factor * (best - particle)

        particle.speed = (omega * particle.speed) + (local_speed_update + global_speed_update)

        # verificando se a nova posição sairá do espaço de busca. Se sim, ajustando para os limites.     
        out_bounds = False
        for i, speed in enumerate(particle.speed):
            if speed > self.max_speed:
                out_bounds = True
                particle.speed[i] = self.max_speed
            if speed < self.min_speed:
                out_bounds = True
                particle.speed[i] = self.min_speed
        
        if out_bounds:
            self.nout_bounds += 1

        # atualizando posição
        particle[:] = particle + particle.speed

    def register_to_update_particles(self):
        self.toolbox.register(alias = 'update',
                              function = self.update_particle)
        
    def criar_registro_geral(self, 
                             nexecucao: int, 
                             func_objetivo: str, 
                             best_particle: list,
                             best_fitness: list,
                             igeneration_stopped: int,
                             exp_path: str):
        d = {
            'execucao': nexecucao,
            'funcao_objetivo': func_objetivo,
            'dimensoes': self.dimensions,
            'tamanho_populacao': self.population_size,
            'total_geracoes_realizadas': igeneration_stopped,
            'range_position': self.bounds,
            'omega': self.omega,
            'reduce_omega_linearly': self.reduce_omega_linearly,
            'reduction_speed_factor': self.reduction_speed_factor,
            'range_speed': [self.min_speed, self.max_speed],
            'cognitive_factor': self.cognitive_update_factor,
            'social_factor': self.social_update_factor,
            'best_particle': best_particle,
            'best_fitness': best_fitness,
            'out_bounds': self.nout_bounds
        }

        self.salvar_registro_geral(registro = d,
                                   exp_path = exp_path)