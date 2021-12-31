import json
import copy
import yaml


def permutate_ga_params(possible_ga_params):
    return [{'mutation_rate': m, 'n_generations': n, 'pop_size': p}
            for m in possible_ga_params['mutation_rate']
            for n in possible_ga_params['n_generations']
            for p in possible_ga_params['pop_size']
            ]


def permutate_pso_params(possible_pso_params):
    return [{'maxiters': i, 'swarm_size': p,
             'acc1': dict_constants['acc1'], 'acc2': dict_constants['acc2'],
             'inertia': dict_constants['inertia']}
            for i in possible_pso_params['maxiters']
            for p in possible_pso_params['swarm_size']
            for dict_constants in possible_pso_params['pso_constants']
            ]


def create_params_clusters(dict_datasets, dict_clusters):
    datasets_clusters_dicts = []

    for dataset_nlabels in dict_datasets:
        nlabels = dataset_nlabels['nlabels']

        for cluster_evals in dict_clusters:
            for i in [1, 2, 3]:
                datasets_clusters_dicts.append(
                    {'fitness': cluster_evals['fitness'],
                     'dataset': dataset_nlabels['dataset'],
                     'lbound': cluster_evals['lbound'],
                     'ubound': cluster_evals['ubound'],
                     'n_clusters': i * nlabels})

    return datasets_clusters_dicts


def create_params_benchmark(dict_benchmarks, particles_lengths):
    params_benchmarks = []

    for benchmark in dict_benchmarks:
        for particle_length in particles_lengths:
            benchmark['particle_length'] = particle_length
            params_benchmarks.append(copy.deepcopy(benchmark))
    return params_benchmarks


def combine_dicts(pso_dict, ga_dict, fitness_dict):
    copy_pso = {p: pso_dict[p] for p in pso_dict}
    copy_pso.update({'lbound': fitness_dict['lbound'], 'ubound': fitness_dict['ubound']})
    if 'particle_length' in fitness_dict:
        copy_pso['particle_length'] = fitness_dict['particle_length']

    combined_dict = {'pso': copy_pso, 'ga': ga_dict, 'fitness': fitness_dict['fitness']}
    if 'dataset' in fitness_dict:
        combined_dict.update({'dataset': fitness_dict['dataset'],
                              'n_clusters': fitness_dict['n_clusters']})
    return combined_dict


def produce_final_configs(params_pso, params_ga, params_fitness):
    return [combine_dicts(pso, ga, fitness)
            for fitness in params_fitness
            for pso in params_pso
            for ga in params_ga]


def save_parameters(parameters_list):
    nfile = 0
    for params in parameters_list:
        nfile += 1
        with open(f'params{nfile}.yml', 'w') as fstream:
            yaml.dump(params, fstream)


def main():
    jsonfile = 'possible_params.json'

    with open(jsonfile, 'r') as stream:
        possible_configs = json.load(stream)
        ga_params = permutate_ga_params(possible_configs['ga'])
        pso_params = permutate_pso_params(possible_configs['pso'])

        benchmark_params = create_params_benchmark(
            possible_configs['benchmark'], possible_configs['particle_length'])

        benchmark_experiments = produce_final_configs(
            pso_params, ga_params, benchmark_params)

        clustering_params = create_params_clusters(
            possible_configs['dataset_nlabels'], possible_configs['cluster_evals'])

        clustering_experiments = produce_final_configs(
            pso_params, ga_params, clustering_params)

        save_parameters(clustering_experiments + benchmark_experiments)


if __name__ == '__main__':
    main()
