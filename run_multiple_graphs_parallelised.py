import multiprocessing
import sys, os
import time
import networkx as nx
import random
import pandas as pd
from src_code.get_data import run_dynamic_adapt_qaoa_silent, run_adapt_qaoa_no_cost_unitaries_silent, run_dynamic_adapt_qaoa_single_mixer_silent, run_adapt_qaoa_silent, run_dynamic_adapt_qaoa_noisy_silent, run_adapt_qaoa_noisy_silent
from src_code import build_operators


def run_algorithm(graph_seed, no_vertices, algo_type, pauli_ops_dict, circuit_depth, output_dir, gate_error_prob=0.0):
    """
    This method creates a graph with the specified number of vertices and random seed.
    It runs the QAOA algorithm, whose type is specified by 'algo_type', and prints the
    optimisation messages to an external text file, as well as the results.
    """

    if not output_dir.endswith('/'):
        output_dir += '/'
    if not os.path.exists(output_dir):
        raise Exception('Error - passed output directory does not exist')
    subfolders = ['convergence_data', 'mixer_gradients', 'output_messages']
    for subfolder in subfolders:
        if not os.path.exists(output_dir + subfolder):
            os.mkdir(output_dir + subfolder)

    graph = nx.Graph()
    edge_list = []
    for node_1 in range(no_vertices):

        for node_2 in range(node_1+1, no_vertices):

            edge_list.append((node_1, node_2))
            
    graph.add_edges_from(edge_list)

    random.seed(graph_seed)
    weights = [random.random() for i in range(len(edge_list))]

    for index, edge in enumerate(graph.edges()):
        graph.get_edge_data(*edge)['weight'] = weights[index]
    
    if algo_type == 'standard':

        pass

    elif algo_type == 'adapt':

        output_file = output_dir + 'output_messages/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.txt'
        gradient_ops_dict = build_operators.build_all_mixers(graph=graph)
        optimisation_results = run_adapt_qaoa_silent(graph, pauli_ops_dict, gradient_ops_dict, max_depth=circuit_depth, output_file=output_file)

        file_name = output_dir + 'convergence_data/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.csv'
        with open(file_name, 'w') as convergence_data_writer:

            convergence_data_writer.write('Layer,Cut Approximation Ratios,Hamiltonian Approximation Ratios,Best Mixer,Best Mixer Param,Best Hamiltonian Param\n')

            for layer in range(len(optimisation_results['cut_approx_ratios'])):
                line = str(layer) + ','
                line += str(optimisation_results['cut_approx_ratios'][layer]) + ','
                line += str(optimisation_results['ham_approx_ratios'][layer]) + ','
                if layer != 0:
                    line += str(optimisation_results['best_mixers'][layer-1]) + ','
                    line += str(optimisation_results['best_mixer_parameters'][layer-1]) + ','
                    line += str(optimisation_results['best_ham_parameters'][layer-1])
                line += '\n'
                convergence_data_writer.write(line)
        
        file_name = output_dir + 'mixer_gradients/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.csv'
        pd.DataFrame(optimisation_results['all_mixers']).to_csv(file_name)

    elif algo_type == 'adapt_noisy':

        output_file = output_dir + 'output_messages/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.txt'
        gradient_ops_dict = build_operators.build_all_mixers(graph=graph)
        optimisation_results = run_adapt_qaoa_noisy_silent(graph, pauli_ops_dict, gradient_ops_dict, max_depth=circuit_depth, output_file=output_file, gate_error_prob=gate_error_prob)

        file_name = output_dir + 'convergence_data/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.csv'
        with open(file_name, 'w') as convergence_data_writer:

            convergence_data_writer.write('Layer,Cut Approximation Ratios,Hamiltonian Approximation Ratios,Best Mixer,Best Mixer Param,Best Hamiltonian Param\n')

            for layer in range(len(optimisation_results['cut_approx_ratios'])):
                line = str(layer) + ','
                line += str(optimisation_results['cut_approx_ratios'][layer]) + ','
                line += str(optimisation_results['ham_approx_ratios'][layer]) + ','
                if layer != 0:
                    line += str(optimisation_results['best_mixers'][layer-1]) + ','
                    line += str(optimisation_results['best_mixer_parameters'][layer-1]) + ','
                    line += str(optimisation_results['best_ham_parameters'][layer-1])
                line += '\n'
                convergence_data_writer.write(line)
        
        file_name = output_dir + 'mixer_gradients/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.csv'
        pd.DataFrame(optimisation_results['all_mixers']).to_csv(file_name)

    elif algo_type == 'dynamic_adapt':

        output_file = output_dir + 'output_messages/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.txt'
        gradient_ops_dict = build_operators.build_all_mixers(graph=graph)
        pauli_mixers_split_ops_dict = build_operators.split_all_mixers(graph)
        optimisation_results = run_dynamic_adapt_qaoa_silent(graph, pauli_ops_dict, gradient_ops_dict, pauli_mixers_split_ops_dict, max_depth=circuit_depth, output_file=output_file)

        file_name = output_dir + 'convergence_data/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.csv'
        with open(file_name, 'w') as convergence_data_writer:

            convergence_data_writer.write('Layer,Cut Approximation Ratios,Hamiltonian Approximation Ratios,Best Mixer,Best Mixer Param,Best Hamiltonian Param\n')

            ham_unitary_count = 0

            for layer in range(len(optimisation_results['cut_approx_ratios'])):
                line = str(layer) + ','
                line += str(optimisation_results['cut_approx_ratios'][layer]) + ','
                line += str(optimisation_results['ham_approx_ratios'][layer]) + ','
                if layer != 0:
                    line += str(optimisation_results['best_mixers'][layer-1]) + ','
                    line += str(optimisation_results['best_mixer_parameters'][layer-1]) + ','
                    if layer in optimisation_results['ham_unitary_layers']:
                        line += str(optimisation_results['best_ham_parameters'][ham_unitary_count])
                        ham_unitary_count += 1
                    else:
                        line += 'nan'
                line += '\n'
                convergence_data_writer.write(line)
        
        file_name = output_dir + 'mixer_gradients/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.csv'
        pd.DataFrame(optimisation_results['all_mixers']).to_csv(file_name)

    elif algo_type == 'dynamic_adapt_noisy':

        output_file = output_dir + 'output_messages/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.txt'
        gradient_ops_dict = build_operators.build_all_mixers(graph=graph)
        pauli_mixers_split_ops_dict = build_operators.split_all_mixers(graph)
        optimisation_results = run_dynamic_adapt_qaoa_noisy_silent(graph, pauli_ops_dict, gradient_ops_dict, pauli_mixers_split_ops_dict, max_depth=circuit_depth, output_file=output_file, gate_error_prob=gate_error_prob)

        file_name = output_dir + 'convergence_data/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.csv'
        with open(file_name, 'w') as convergence_data_writer:

            convergence_data_writer.write('Layer,Cut Approximation Ratios,Hamiltonian Approximation Ratios,Best Mixer,Best Mixer Param,Best Hamiltonian Param\n')

            ham_unitary_count = 0

            for layer in range(len(optimisation_results['cut_approx_ratios'])):
                line = str(layer) + ','
                line += str(optimisation_results['cut_approx_ratios'][layer]) + ','
                line += str(optimisation_results['ham_approx_ratios'][layer]) + ','
                if layer != 0:
                    line += str(optimisation_results['best_mixers'][layer-1]) + ','
                    line += str(optimisation_results['best_mixer_parameters'][layer-1]) + ','
                    if layer in optimisation_results['ham_unitary_layers']:
                        line += str(optimisation_results['best_ham_parameters'][ham_unitary_count])
                        ham_unitary_count += 1
                    else:
                        line += 'nan'
                line += '\n'
                convergence_data_writer.write(line)
        
        file_name = output_dir + 'mixer_gradients/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.csv'
        pd.DataFrame(optimisation_results['all_mixers']).to_csv(file_name)


    elif algo_type == 'dynamic_adapt_no_cost':

        output_file = output_dir + 'output_messages/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.txt'
        gradient_ops_dict = build_operators.build_all_mixers(graph=graph)
        optimisation_results = run_adapt_qaoa_no_cost_unitaries_silent(graph, pauli_ops_dict, gradient_ops_dict, max_depth=circuit_depth, output_file=output_file)

        file_name = output_dir + 'convergence_data/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.csv'
        with open(file_name, 'w') as convergence_data_writer:

            convergence_data_writer.write('Layer,Cut Approximation Ratios,Hamiltonian Approximation Ratios,Best Mixer,Best Mixer Param,Best Hamiltonian Param\n')

            ham_unitary_count = 0

            for layer in range(len(optimisation_results['cut_approx_ratios'])):
                line = str(layer) + ','
                line += str(optimisation_results['cut_approx_ratios'][layer]) + ','
                line += str(optimisation_results['ham_approx_ratios'][layer]) + ','
                if layer != 0:
                    line += str(optimisation_results['best_mixers'][layer-1]) + ','
                    line += str(optimisation_results['best_mixer_parameters'][layer-1]) + ','
                    if layer in optimisation_results['ham_unitary_layers']:
                        line += str(optimisation_results['best_ham_parameters'][ham_unitary_count])
                        ham_unitary_count += 1
                    else:
                        line += 'nan'
                line += '\n'
                convergence_data_writer.write(line)
        
        file_name = output_dir + 'mixer_gradients/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.csv'
        pd.DataFrame(optimisation_results['all_mixers']).to_csv(file_name)

    elif algo_type == 'dynamic_adapt_zero_gamma':

        output_file = output_dir + 'output_messages/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.txt'
        gradient_ops_dict = build_operators.build_all_mixers(graph=graph)
        pauli_mixers_split_ops_dict = build_operators.split_all_mixers(graph)
        optimisation_results = run_dynamic_adapt_qaoa_single_mixer_silent(graph, pauli_ops_dict, gradient_ops_dict, pauli_mixers_split_ops_dict, max_depth=circuit_depth, output_file=output_file)

        file_name = output_dir + 'convergence_data/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.csv'
        with open(file_name, 'w') as convergence_data_writer:

            convergence_data_writer.write('Layer,Cut Approximation Ratios,Hamiltonian Approximation Ratios,Best Mixer,Best Mixer Param,Best Hamiltonian Param\n')

            ham_unitary_count = 0

            for layer in range(len(optimisation_results['cut_approx_ratios'])):
                line = str(layer) + ','
                line += str(optimisation_results['cut_approx_ratios'][layer]) + ','
                line += str(optimisation_results['ham_approx_ratios'][layer]) + ','
                if layer != 0:
                    line += str(optimisation_results['best_mixers'][layer-1]) + ','
                    line += str(optimisation_results['best_mixer_parameters'][layer-1]) + ','
                    if layer in optimisation_results['ham_unitary_layers']:
                        line += str(optimisation_results['best_ham_parameters'][ham_unitary_count])
                        ham_unitary_count += 1
                    else:
                        line += 'nan'
                line += '\n'
                convergence_data_writer.write(line)
        
        file_name = output_dir + 'mixer_gradients/graph_' + str(no_vertices) + '_nodes_seed_' + str(graph_seed) + '.csv'
        pd.DataFrame(optimisation_results['all_mixers']).to_csv(file_name)

    else:
        raise Exception('Error - invalid argument type passed!')
    

if __name__ == '__main__':

    starttime = time.time()

    arguments = sys.argv
    if len(arguments) != 7 and len(arguments) != 8:
        raise Exception
    no_vertices = int(arguments[1])
    no_graphs = int(arguments[2])
    init_seed = int(arguments[3])
    max_depth = int(arguments[4])
    algo_type = arguments[5]
    output_dir = arguments[6]
    if not output_dir.endswith('/'):
        output_dir = output_dir + '/' + algo_type + '/'
    else:
        output_dir = output_dir + algo_type + '/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if 'noisy' in algo_type:
        gate_error_prob = float(arguments[7])
    else:
        gate_error_prob=0.0

    pauli_ops_dict = build_operators.build_all_paulis(no_vertices)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()//2)

    arguments = [None] * no_graphs
    for i in range(no_graphs):
        arguments[i] = [i+init_seed, no_vertices, algo_type, pauli_ops_dict, max_depth, output_dir, gate_error_prob]

    pool.starmap(run_algorithm, arguments)
    pool.close()

    print('This took {} seconds'.format(time.time()-starttime))