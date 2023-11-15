import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from classes.base.namespaces import ExperimentDict
from classes.helper.utils import Utils

from deap import base, creator, tools

class PSO:
    def __init__(self,
                 exp_dict: ExperimentDict,
                 paths_dict: dict,
                 show_log: bool = True):
        
        self.population_size = exp_dict.heuristic_opt.population_size
        self.l_bound = exp_dict.heuristic_opt.l_bound
        self.u_bound = exp_dict.heuristic_opt.u_bound
        self.omega = exp_dict.heuristic_opt.omega
        self.min_speed = exp_dict.heuristic_opt.min_speed
        self.max_speed = exp_dict.heuristic_opt.max_speed
        self.cognitive_update_factor = exp_dict.heuristic_opt.cognitive_update_factor
        self.social_update_factor = exp_dict.heuristic_opt.social_update_factor
        self.reduce_omega_linearly = exp_dict.heuristic_opt.reduce_omega_linearly
        self.reduction_speed_factor = exp_dict.heuristic_opt.reduction_speed_factor
        
        self.paths_dict = paths_dict
        self.show_log = show_log

        self.nparams = len(self.u_bound)

        self.max_evaluations = exp_dict.heuristic_opt.max_evaluations
        self.nout_bounds = 0
        self.best = None
        self.toolbox = base.Toolbox()

    def main(self, 
             func_,
             nexecucao: int = 0):
        
        self.toolbox.register(alias = 'evaluate', 
                              function = func_)
        
        ## inicializações
        self.define_as_minimization_problem()
        self.creating_particle_class()
        self.create_particle()
        self.creating_particle_register()
        self.creating_population_register()
        self.register_to_update_particles()

        ## inicializando front de pareto
        pareto = tools.ParetoFront()

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
            print(f"### Geração {idx} ###")
            # reduzindo omega linearmente
            if self.reduce_omega_linearly:
                omega = self.omega - (idx * (self.omega - 0.4) / (self.max_evaluations * self.reduction_speed_factor))
            
            # avaliar todas as partículas na população
            for particle in population:
                # calcular o valor de fitness da partícula / avaliação
                particle.fitness.values = self.toolbox.evaluate(particle)
                # atualizando melhor partícula global
                if particle.best is None or particle.best.size == 0 or particle.best.fitness < particle.fitness:
                    particle.best = creator.Particle(particle)
                    particle.best.fitness.values = particle.fitness.values
                # atualizando valor global
                if best is None or best.size == 0 or best.fitness < particle.fitness:
                    best = creator.Particle(particle)
                    best.fitness.values = particle.fitness.values
                # atualizando número de avaliações, verificando limites de tempo de otimização permitidos
                # nevaluations += 1
                # stop_cond1 = abs(best.fitness.values[0] - func_[1]) < 10e-8
                # stop_cond2 = nevaluations >= self.max_evaluations
                # if stop_cond1 or stop_cond2:
                #     finish_optimization = True
                #     igeneration_stopped = idx
                #     break

            # atualizando velocidade e posição
            for particle in population:
                self.toolbox.update(particle, best, omega)

            avg_euclidian_distance_history.append(self.calc_distancia_euclidiana_media_da_populacao(population))
            # salvando as estatísticas
            # if generation == 1 or generation % 500 == 0:
            best_fitness_history.append(best.fitness.values)
            logbook.record(gen = generation,
                           evals = len(population),
                           **stats.compile(population))
            if self.show_log:
                if self.reduce_omega_linearly:
                    print(logbook.stream + f" | omega = {omega}")
                else:
                    print(logbook.stream)

        avg_fitness_history = [logbook[i]['avg'] for i in range(len(logbook))]
        self.best = best
        pareto.update(population)
        
        self.print_informacoes_gerais_optimizacao(best)
        self.criar_grafico_fronte_de_pareto(pareto_front = pareto,
                                            imgs_path = self.paths_dict['eval_imgs'], 
                                            img_name = f"pso_pareto_front_{nexecucao}")
        
        self.criar_grafico_evolucao_fitness(hist_best_fitness = best_fitness_history,
                                            hist_avg_fitness = avg_fitness_history,
                                            imgs_path = self.paths_dict['eval_imgs'], 
                                            img_name = f"pso_exec_{nexecucao}",
                                            use_log = True)
        self.criar_grafico_evolucao_distancia_media_pontos(hist_dist_pontos = avg_euclidian_distance_history,
                                                           imgs_path = self.paths_dict['eval_imgs'],
                                                           img_name = f"pso_distance_particles_{nexecucao}")
        self.criar_registro_geral(nexecucao = nexecucao,
                                  best_particle = best,
                                  best_fitness = best.fitness.values[0],
                                #   igeneration_stopped = igeneration_stopped,
                                  exp_path = self.paths_dict['eval_metrics'])
        
        self.salvar_historico_em_arquivo_txt([ind.fitness.values for ind in pareto], self.paths_dict['eval_metrics'], f'pareto_front_{nexecucao}')
        self.salvar_historico_em_arquivo_txt(best_fitness_history, self.paths_dict['eval_metrics'], f'best_fitness_{nexecucao}')
        self.salvar_historico_em_arquivo_txt(avg_fitness_history, self.paths_dict['eval_metrics'], f'avg_fitness_{nexecucao}')
        self.salvar_historico_em_arquivo_txt(avg_euclidian_distance_history, self.paths_dict['eval_metrics'], f'dist_particles_{nexecucao}')

        del creator.FitnessMin
        del creator.Particle

    def print_informacoes_gerais_optimizacao(self, best):
        print("-- Melhor partícula = ", best)
        print("-- Melhor fitness = ", best.fitness.values[0])
        # print("-- Geração que parou a otimização = ", igeneration_stopped)
        print("-- Qtd de vezes que saiu do espaço de busca = ", self.nout_bounds)

    def define_as_minimization_problem(self):
        creator.create(name = "FitnessMin",
                       base = base.Fitness,
                       weights = (-1., -1.))
        
    def creating_particle_class(self):
        creator.create(name = 'Particle',
                       base = np.ndarray,
                       fitness = creator.FitnessMin,
                       speed = None,
                       best = None)
        
    def creating_individual_attr_register(self):
        for i in range(self.nparams):
            self.toolbox.register("attribute_" + str(i),
                                  np.random.uniform,
                                  self.l_bound[i],
                                  self.u_bound[i])
    
    def get_individual_attr(self):
        attr = ()
        for i in range(self.nparams):
            attr = attr + (self.toolbox.__getattribute__("attribute_" + str(i)),)
        
        return attr
    
    # def creating_particle_register_v2(self):
    #     self.creating_individual_attr_register()
    #     attr = self.get_individual_attr

    #     self.toolbox.register("particleCreator",
    #                           tools.initCycle,
    #                           creator.Particle,
    #                           attr,
    #                           n = 1)
        
    def create_particle(self):
        arr_values = np.zeros(self.nparams)
        for i in range(self.nparams):
            arr_values[i] = np.random.uniform(low = self.l_bound[i],
                                              high = self.u_bound[i],
                                              size = 1)

        particle = creator.Particle(arr_values)
        
        particle.speed = np.random.uniform(low = self.min_speed,
                                           high = self.max_speed,
                                           size = len(self.u_bound))
        
        return particle
    
    def creating_particle_register(self):
        self.toolbox.register(alias = 'particleCreator',
                              function = self.create_particle)

    def creating_population_register(self):      
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
        nposition = particle + particle.speed
        for i, pos in enumerate(nposition):
            if pos > self.u_bound[i]:
                nposition[i] = self.u_bound[i]
            if pos < self.l_bound[i]:
                nposition[i] = self.l_bound[i]

        particle[:] = nposition

    def register_to_update_particles(self):
        self.toolbox.register(alias = 'update',
                              function = self.update_particle)
        
    def criar_registro_geral(self, 
                             nexecucao: int,  
                             best_particle: list,
                             best_fitness: list,
                            #  igeneration_stopped: int,
                             exp_path: str):
        d = {
            'execucao': nexecucao,
            'tamanho_populacao': self.population_size,
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
        
    def calc_distancia_euclidiana_media_da_populacao(self, population):
        # Inicializando uma matriz para armazenar as distâncias
        num_vectors = len(population)
        distances = np.zeros((num_vectors, num_vectors))

        # Calcula as distâncias euclidianas entre os vetores
        for i in range(num_vectors):
            for j in range(i, num_vectors):
                distance = np.sqrt(np.sum((population[i] - population[j])**2))
                distances[i, j] = distance
                distances[j, i] = distance

        # Calcula a média das distâncias
        average_distance = np.mean(distances)

        return average_distance
    
    def salvar_registro_geral(self,
                              registro: dict,
                              exp_path: str):
        
        df_registro = pd.DataFrame([registro])
        Utils.save_experiment_as_csv(base_dir = exp_path, dataframe = df_registro, filename = 'opt_history')

    def salvar_historico_em_arquivo_txt(self, lista: list, caminho_do_arquivo: str, nome_do_arquivo: str):
        Utils.save_list_as_txt(lista, caminho_do_arquivo, nome_do_arquivo)

    def criar_grafico_evolucao_fitness(self, 
                                       hist_best_fitness: list, 
                                       hist_avg_fitness: list, 
                                       imgs_path: str, 
                                       img_name: str,
                                       use_log: bool = False) -> None:
        
        plt.figure()
        xticks_ajustado = [v * 500 for v in range(len(hist_best_fitness))]
        
        plt.plot(hist_best_fitness, color = 'green')
        plt.plot(hist_avg_fitness, color = 'red')
        
        plt.title('Best e Avg fitness através das gerações')
        plt.xlabel('Gerações')
        plt.ylabel('Fitness')
        plt.legend(['Best', 'Avg'])
        
        if use_log:
            plt.yscale('log')

        filename = f'{imgs_path}/{img_name}.jpg' 
        plt.savefig(filename)

    def criar_grafico_fronte_de_pareto(self, 
                                       pareto_front, 
                                       imgs_path: str, 
                                       img_name: str):
        
        obj1, obj2 = zip(*[ind.fitness.values for ind in pareto_front])

        plt.figure()
        plt.scatter(x = obj2, y = obj1, c='black', marker='+')
        
        plt.title('Front de Pareto')
        plt.xlabel('Func obj - Topology')
        plt.ylabel('Func obj - KID')

        filename = f'{imgs_path}/{img_name}.jpg' 
        plt.savefig(filename)


    def criar_grafico_evolucao_distancia_media_pontos(self, 
                                                      hist_dist_pontos: list,
                                                      imgs_path: str, 
                                                      img_name: str,
                                                      use_log: bool = False) -> None:
        
        plt.figure()
        
        plt.plot(hist_dist_pontos, color = 'green')
        
        plt.title('Distância média das partículas')
        plt.xlabel('Gerações')
        plt.ylabel('Avg')
        plt.legend(['Avg'])
        
        if use_log:
            plt.yscale('log')

        filename = f'{imgs_path}/{img_name}.jpg' 
        plt.savefig(filename)