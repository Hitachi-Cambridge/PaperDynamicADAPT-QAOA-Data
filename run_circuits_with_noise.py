# python script which builds adapt qaoa circuits from passed directories with
# layer data and runs them in a noisy setting

import networkx as nx
import sys
sys.path.append('..')
import os
import random
from src_code import get_data
from src_code import build_operators
from src_code import useful_methods
import pandas as pd
import numpy as np
import time
import multiprocessing

def simulate_circuit(data_dir, gate_error_prob, pauli_ops_dict, approx_ratios_dataframe):

    max_no_layers = 0

    no_nodes = int(data_dir.split('/')[-1].split('_')[1])
    seed = int(data_dir.split('/')[-1].split('_')[-1].split('.')[0])

    # build random graph
    graph = nx.Graph()
    edge_list = []
    for node_1 in range(no_nodes):

        for node_2 in range(node_1+1, no_nodes):

            edge_list.append((node_1, node_2))
            
    graph.add_edges_from(edge_list)

    random.seed(seed)
    weights = [random.random() for i in range(len(edge_list))]

    for index, edge in enumerate(graph.edges()):
        graph.get_edge_data(*edge)['weight'] = weights[index]

    # run noisy circuit

    ham = build_operators.cut_hamiltonian(graph)
    optimal_solution = useful_methods.find_optimal_cut(graph)
    max_cut_value = optimal_solution[1]
    max_ham_value = optimal_solution[2]
    ham_offset = max_cut_value - max_ham_value

    noiseless_data = pd.read_csv(data_dir, header=0)
    mixer_types = noiseless_data['Best Mixer'].dropna().to_list()
    mixer_params = noiseless_data['Best Mixer Param'].dropna().to_list()
    ham_params = noiseless_data['Best Hamiltonian Param'].dropna().to_list()
    ham_layers = noiseless_data['Best Hamiltonian Param'].dropna().index.to_list()
    noiseless_approx_ratios = noiseless_data['Cut Approximation Ratios'].to_list()
    no_layers = len(mixer_params)
    if no_layers > max_no_layers:
        max_no_layers = no_layers

    if gate_error_prob == 0.0:
        approx_ratios_dataframe['Seed ' + str(seed)] = noiseless_approx_ratios
        return

    ham_unitaries_count = 0
    noisy_approx_ratios = []

    noisy_dens_mat = build_operators.initial_density_matrix(no_nodes)
    curr_ham_estimate = (noisy_dens_mat * ham).trace().real
    curr_cut_approx = (curr_ham_estimate + ham_offset) / max_cut_value
    noisy_approx_ratios.append(curr_cut_approx)

    for layer in range(no_layers):

        # check if we should add Hamiltonian unitary in current layer
        if len(ham_layers) > ham_unitaries_count and ham_layers[ham_unitaries_count] == layer + 1:

            cut_unit = build_operators.cut_unitary(graph, ham_params[ham_unitaries_count], dict_paulis=pauli_ops_dict)
            noisy_dens_mat = (cut_unit * noisy_dens_mat) * (cut_unit.transpose().conj())
            ham_unitaries_count += 1
            # apply noise
            noisy_dens_mat = useful_methods.noisy_ham_unitary_evolution(noisy_dens_mat, noise_prob=gate_error_prob, graph=graph, pauli_dict=pauli_ops_dict)

        mix_unit = build_operators.mixer_unitary(mixer_types[layer], mixer_params[layer], dict_paulis=pauli_ops_dict, no_nodes=no_nodes)
        noisy_dens_mat = (mix_unit * noisy_dens_mat) * (mix_unit.transpose().conj())
        # apply noise
        noisy_dens_mat = useful_methods.noisy_mixer_unitary_evolution(noisy_dens_mat, gate_error_prob, mixer_types[layer], pauli_dict=pauli_ops_dict)

        curr_ham_estimate = (noisy_dens_mat * ham).trace().real
        curr_cut_approx = (curr_ham_estimate + ham_offset) / max_cut_value
        noisy_approx_ratios.append(curr_cut_approx)

    approx_ratios_dataframe['Seed ' + str(seed)] = noisy_approx_ratios


if __name__ == '__main__':

    starttime = time.time()

    arguments = sys.argv
    if len(arguments) != 3:
        raise Exception
    data_dir = arguments[1]
    gate_error_prob = float(arguments[2])
    if not data_dir.endswith('/'):
        data_dir = data_dir + '/' + 'convergence_data/'
    else:
        data_dir = data_dir + 'convergence_data/'

    for file in os.listdir(data_dir):
        if 'DS_Store' in file:
            continue
        no_vertices = int(file.split('_')[1])
        break

    pauli_ops_dict = build_operators.build_all_paulis(no_vertices)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count()//2)

    approx_ratios_df = multiprocessing.Manager().dict()
    arguments = []
    for file in os.listdir(data_dir):
        if 'DS_Store' in file:
            continue
        arguments.append([data_dir + file, gate_error_prob, pauli_ops_dict, approx_ratios_df])

    pool.starmap(simulate_circuit, arguments)
    pool.close()

    approx_ratios_df = dict(approx_ratios_df)
    approx_ratios_df = pd.DataFrame(approx_ratios_df)
    output_dir = data_dir + '../noisy_data/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    approx_ratios_df.to_csv(output_dir + '_' + str(gate_error_prob) + '_gate_error_prob.csv')

    print('This took {} seconds'.format(time.time()-starttime))