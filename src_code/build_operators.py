"""
Python file containing methods which build operators and ansätze
"""

from qiskit import quantum_info as qi
from networkx import Graph
import numpy as np
import warnings
import networkx as nx
from src_code.mixers_density import *
from scipy import sparse
import math
from src_code import useful_methods

def initial_density_matrix(no_qubits):
    """
    Returns density matrix corresponding to the initial state for QAOA algorithms.

    Parameters:
        no_qubits - number of qubits in system
    
    Returns:
        dens_mat - DensityMatrix Instance
    """

    dim = 2**no_qubits
    dens_mat = qi.DensityMatrix(np.full((dim, dim), 1/dim))

    return sparse.csr_matrix(dens_mat.data)

def cut_hamiltonian(graph):
    """
    Returns cut Hamiltonian operator.
    """
    if not isinstance(graph, Graph):
        raise Exception("Error - passed graph must be instance of networkx Graph class!")

    hamiltonian_operator = None
    no_nodes = graph.number_of_nodes()

    with warnings.catch_warnings():

        warnings.simplefilter("ignore", category=FutureWarning)
        no_ops = graph.number_of_edges()

    pauli_strings = [None] * no_ops
    coeffs = [None] * no_ops
    index = 0

    for i in range(no_nodes):

        for k in range(i+1, no_nodes):

            if graph.get_edge_data(i, k) != None:
                
                tmp_str = 'I' * (i) + 'Z' + 'I' * (k-i-1) + 'Z' + 'I' * (no_nodes - k - 1)
                tmp_str = tmp_str[::-1]
                pauli_strings[index] = tmp_str
                coeffs[index] = (-0.5) * graph.get_edge_data(i, k)['weight']
                index += 1

    hamiltonian_operator = qi.SparsePauliOp(pauli_strings, np.array(coeffs)).to_operator()

    return sparse.csr_matrix(hamiltonian_operator.data)

def cut_unitary(graph, parameter, dict_paulis):
    """
    Returns unitary operator corresponding to exponential of cut Hamiltonian.
    """

    if not isinstance(graph, nx.Graph):
        raise Exception

    first = True
    for edge in graph.edges:

        weight = graph.get_edge_data(*edge)['weight']
        total_param = 0.5 * parameter * weight
        key = 'Z' + str(edge[0]) + 'Z' + str(edge[1])

        if key not in dict_paulis:
            key = 'Z' + str(edge[1]) + 'Z' + str(edge[0])
        if key not in dict_paulis:
            raise Exception
        
        tmp_matrix = dict_paulis['I'] * math.cos(total_param) + dict_paulis[key] * math.sin(total_param) * 1j

        if first:
            result = tmp_matrix
            first = False
        else:
            result = tmp_matrix * result

    return result

def mixer_unitary(mixer_type, parameter_value, dict_paulis, no_nodes):
    """
    Returns unitary operator corresponding to expontential of mixer of specified type.
    """
    if mixer_type == 'standard_x' or mixer_type == 'standard_y':

        first = True
        for i in range(no_nodes):

            if first:

                result = math.cos(parameter_value) * dict_paulis['I'] - 1j * math.sin(parameter_value) * dict_paulis[mixer_type[-1].upper() + str(i)]
                first = False

            else:

                result = result * (math.cos(parameter_value) * dict_paulis['I'] - 1j * math.sin(parameter_value) * dict_paulis[mixer_type[-1].upper()  + str(i)])

    else:

        result = math.cos(parameter_value) * dict_paulis['I'] - 1j * math.sin(parameter_value) * dict_paulis[mixer_type]

    return result

def build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_dict, ham_layers = None, noisy=False, noise_prob=0.0):
    """
    Returns the density matrix of a (Dynamic) ADAPT-QAOA circuit.

    Parameters:
        graph - Networkx Graph instance of graph on which to run algorithm
        mixer_params - list of parameters to use for mixer unitaries
        mixer_list - list of mixer types to use
        ham_params - list of parameters to use for Hamiltonian unitaries
        pauli_dict - dictionary of Pauli sparse matrices
        ham_layers - list of layers in which to place a Hamiltonian unitary (default is None and corresponds to the usual ADAPT-QAOA ansatz)
        noisy - boolean variable denoting whether the circuit should be built in a noisy setting (default if False)
        noise_prob - what gate-error probability to use (default is 0.0 corresponding to no noise)

    Output:
        dens_mat - density matrix output of ansatz as instance of scipy.sparse csr_matrix class
    """

    if not isinstance(graph, Graph):
        raise Exception("Error - passed graph is not instance of Networkx Graph class!")

    no_layers = len(mixer_params)

    if no_layers != len(mixer_list):
        raise Exception('Error - incompatible number of mixer types and mixer unitary parameters were passed!')

    if ham_layers == None:
        ham_layers = [i+1 for i in range(no_layers)]

    if len(ham_params) != len(ham_layers):
        raise Exception("Error - incompatible number of Hamiltonian mixer parameters and layer numbers were passed!")
    
    if noisy and noise_prob == 0.0:
        raise Exception('Error - to execute a noisy circuit, one must pass a non-zero gate-error probability!')
    
    no_qubits = graph.number_of_nodes()
    dens_mat = initial_density_matrix(no_qubits)

    ham_unitaries_count = 0
    
    for layer in range(no_layers):

        if len(ham_layers) > ham_unitaries_count and ham_layers[ham_unitaries_count] == layer + 1:

            cut_unit = cut_unitary(graph, ham_params[ham_unitaries_count], dict_paulis=pauli_dict)
            dens_mat = (cut_unit * dens_mat) * (cut_unit.transpose().conj())
            ham_unitaries_count += 1
            if noisy:
                dens_mat = useful_methods.noisy_ham_unitary_evolution(dens_mat, noise_prob=noise_prob, graph=graph, pauli_dict=pauli_dict)

        mix_unit = mixer_unitary(mixer_list[layer], mixer_params[layer], dict_paulis=pauli_dict, no_nodes=no_qubits)
        dens_mat = (mix_unit * dens_mat) * (mix_unit.transpose().conj())
        if noisy:
            dens_mat = useful_methods.noisy_mixer_unitary_evolution(dens_mat, noise_prob, mixer_list[layer], pauli_dict=pauli_dict)

    return dens_mat

def build_standard_qaoa_ansatz(graph, parameter_list, pauli_dict, noisy=False, noise_prob=0.0):
    """
    Returns density matrix of Standard QAOA circuit.

    Parameters:
        graph - Networkx Graph instance of graph on which to run algorithm.
        parameter_list - list of parameter values for ansatz
        noise - use noise models or not

    Returns:
        dens_mat - Instance of DensityMatrix class.
    """

    if not isinstance(graph, Graph):
        raise Exception("Error - passed graph is not instance of Networkx Graph class!")

    no_layers = len(parameter_list) // 2

    ham_parameters = parameter_list[:no_layers]
    mixer_parameters = parameter_list[no_layers:]
    
    no_qubits = graph.number_of_nodes()

    dens_mat = initial_density_matrix(no_qubits)
    
    for layer in range(no_layers):

        cut_unit = cut_unitary(graph, ham_parameters[layer], pauli_dict)
        dens_mat = (cut_unit * dens_mat) * (cut_unit.transpose().conj())
        if noisy:
            dens_mat = useful_methods.noisy_ham_unitary_evolution(dens_mat, noise_prob=noise_prob, graph=graph, pauli_dict=pauli_dict)
    
        mix_unit = mixer_unitary('standard_x', mixer_parameters[layer], pauli_dict, no_qubits)
        dens_mat = (mix_unit * dens_mat) * (mix_unit.transpose().conj())

    return dens_mat

def build_all_mixers(graph):
    """
    Method which builds all possible mixers for the passed graph,
    and stores them in a dictionary.
    """
    if not isinstance(graph, Graph):
        raise Exception("Error - passed graph is not an instance of the networkx Graph class!")

    dict_mixers = {}
    no_qubits = graph.number_of_nodes()

    single_qubit_mixers = ["X", "Y"]
    double_qubit_mixers = ["XZ", "YZ", "XY", "XX", "YY"]

    for mixer_type in single_qubit_mixers:

        for qubit in range(no_qubits):

            key = mixer_type + str(qubit)
            if mixer_type == 'X':
                dict_mixers[key] = X_mixer(graph, qubit)
            elif mixer_type == 'Y':
                dict_mixers[key] = Y_mixer(graph, qubit)

    for mixer_type in double_qubit_mixers:

        for qubit_1 in range(no_qubits):
            
            for qubit_2 in range(no_qubits):

                if qubit_1 == qubit_2:
                    continue

                key = mixer_type[0] + str(qubit_1) + mixer_type[1] + str(qubit_2)

                if mixer_type == 'XZ':
                    dict_mixers[key] = XZ_mixer(graph, qubit_1, qubit_2)
                if mixer_type == 'YZ':
                    dict_mixers[key] = YZ_mixer(graph, qubit_1, qubit_2)
                if mixer_type == 'XY':
                    dict_mixers[key] = XY_mixer(graph, qubit_1, qubit_2)
                if qubit_2 > qubit_1:
                    if mixer_type == 'XX':
                        dict_mixers[key] = XX_mixer(graph, qubit_1, qubit_2)
                    if mixer_type == 'YY':
                        dict_mixers[key] = YY_mixer(graph, qubit_1, qubit_2)

    return dict_mixers

def split_all_mixers(graph):
    """
    A method which returns a dictionary containing the commuting and anti-commuting
    parts of the Ising Hamiltonian for all possible mixers (except the standard ones)
    for which such a splitting is not possible.
    """

    if not isinstance(graph, Graph):
        raise Exception("Error - passed graph is not an instance of the networkx Graph class!")

    result = {}
    no_qubits = graph.number_of_nodes()

    single_qubit_mixers = ["X", "Y"]
    double_qubit_mixers = ["XZ", "YZ", "XY", "XX", "YY"]

    for mixer_type in single_qubit_mixers:

        for qubit in range(no_qubits):

            key = mixer_type + str(qubit)
            H_c, H_a = useful_methods.split_hamiltonian(graph, key)
            result[key] = {
                'H_c' : H_c,
                'H_a' : H_a
            }

    for mixer_type in double_qubit_mixers:

        for qubit_1 in range(no_qubits):
            
            for qubit_2 in range(no_qubits):

                if qubit_1 == qubit_2:
                    continue
                if (mixer_type == 'XX' or mixer_type == 'YY') and qubit_1 > qubit_2:
                    continue

                key = mixer_type[0] + str(qubit_1) + mixer_type[1] + str(qubit_2)
                H_c, H_a = useful_methods.split_hamiltonian(graph, key)
                result[key] = {
                    'H_c' : H_c,
                    'H_a' : H_a
                }
                
    return result

def build_all_paulis(no_nodes):
    """
    A method which builds all the possible Pauli matrices appearing
    in the unitary building blocks of the circuits and returns them
    in the form of sparse density matrices stored in a dictionary.
    """
    result = {}

    mixer_types = ['X', 'Y', 'Z', 'XX', 'YY', 'ZZ', 'XZ', 'YZ', 'XY']
    for mixer in mixer_types:

        if len(mixer) == 1:

            for node in range(no_nodes):

                key = mixer + str(node)
                pauli_string = 'I' * (node) + mixer + 'I' * (no_nodes-node-1)
                pauli_string = pauli_string[::-1]
                result[key] = sparse.csr_matrix(qi.Pauli(pauli_string).to_matrix())

        if len(mixer) == 2:

            for node_1 in range(no_nodes):

                for node_2 in range(no_nodes):

                    if node_1 == node_2:
                        continue

                    if mixer[0] == mixer[1] and node_2 < node_1:
                        continue

                    key = mixer[0] + str(node_1) + mixer[1] + str(node_2)
                    if node_1 > node_2:
                        larger_node = node_1
                        larger_type = mixer[0]
                        smaller_node = node_2
                        smaller_type = mixer[1]
                    else:
                        larger_node = node_2
                        larger_type = mixer[1]
                        smaller_node = node_1
                        smaller_type = mixer[0]
                    pauli_string = 'I' * (smaller_node) + smaller_type + 'I' * (larger_node-smaller_node-1) + larger_type + 'I' * (no_nodes - larger_node - 1)
                    pauli_string = pauli_string[::-1]
                    result[key] = sparse.csr_matrix(qi.Pauli(pauli_string).to_matrix())

    result['I'] = sparse.csr_matrix(np.identity(2**no_nodes, dtype=complex))

    return result

    
