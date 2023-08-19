"""
Python file containing useful methods
"""

from qiskit import quantum_info as qi
import networkx as nx
from src_code import build_operators
import warnings
from scipy import sparse, linalg
import cvxpy as cvx
import numpy as np


# def find_best_mixer(dens_mat, graph, gamma_0=0.01):
#     """
#     Finds the mixer operator with the largest energy gradient.

#     Parameters:
#         dens_mat - DensityMatrix instance for which to find the energy gradients
#         graph - Graph instance corresponding to problem at hand
#         gamma_0 - parameter for cut hamiltonian unitary
#     Returns:
#         sorted_mixers - sorted list of tuples containing mixer type and absolute value of its
#         gradient
#     """

#     if not isinstance(graph, Graph):
#         raise Exception("Error - passed graph must be instance of networkx Graph class!")

#     if not isinstance(dens_mat, qi.DensityMatrix):
#         raise Exception("Error - passed density matrix is not DensityMatrix instance!")

#     if graph.number_of_nodes() != dens_mat.num_qubits:
#         raise Exception("Error - number of nodes in graph does not agree with number of qubits in quantum circuit!")

#     no_qubits = dens_mat.num_qubits
#     new_dens_mat = dens_mat.evolve(cut_unitary(graph, gamma_0))

#     single_qubit_mixers = ["X", "Y"]
#     double_qubit_mixers = ["XZ", "YZ", "XY", "XX", "YY"]
#     standard_mixers = ["standard_x", "standard_y"]

#     dict_mixers = {}

#     for mixer_type in single_qubit_mixers:

#             for qubit in range(no_qubits):

#                 key = mixer_type + str(qubit)

#                 if mixer_type == "X":
#                     tmp_mixer = X_mixer(no_qubits, qubit)
#                 elif mixer_type == "Y":
#                     tmp_mixer = Y_mixer(no_qubits, qubit)
#                 else:
#                     raise Exception

#                 # print("Finding gradient for mixer", key)

#                 tmp_mixer.find_exact_gradient(new_dens_mat, graph)
#                 dict_mixers[key] = tmp_mixer.mean_gradient

#     for mixer_type in double_qubit_mixers:

#             for qubit_1 in range(no_qubits):

#                 for qubit_2 in range(no_qubits):

#                     if qubit_1 == qubit_2:
#                         continue
#                     if (mixer_type == "XX" or mixer_type == "YY") and qubit_2 < qubit_1:
#                         continue

#                     key = mixer_type[0] + str(qubit_1) + mixer_type[1] + str(qubit_2)

#                     if mixer_type == "XZ":
#                         tmp_mixer = XZ_mixer(no_qubits, qubit_1, qubit_2)
#                     elif mixer_type == "YZ":
#                         tmp_mixer = YZ_mixer(no_qubits, qubit_1, qubit_2)
#                     elif mixer_type == "XY":
#                         tmp_mixer = XY_mixer(no_qubits, qubit_1, qubit_2)
#                     elif mixer_type == "XX":
#                         tmp_mixer = XX_mixer(no_qubits, qubit_1, qubit_2)
#                     elif mixer_type == "YY":
#                         tmp_mixer = YY_mixer(no_qubits, qubit_1, qubit_2)

#                     # print("Finding gradient for mixer", key)
#                     tmp_mixer.find_exact_gradient(new_dens_mat, graph)
#                     dict_mixers[key] = tmp_mixer.mean_gradient

#     for mixer_type in standard_mixers:

#         key = mixer_type

#         if mixer_type[-1] == "x":
#             tmp_mixer = standard_mixer(no_qubits)
#         elif mixer_type[-1] == "y":
#             tmp_mixer = standard_mixer_y_gates(no_qubits)

#         # print("Finding gradient for mixer", key)
#         tmp_mixer.find_exact_gradient(new_dens_mat, graph)
#         dict_mixers[key] = tmp_mixer.mean_gradient

#     # sort gradients by absolute value and find best mixer
#     for key in dict_mixers:
#         dict_mixers[key] = abs(dict_mixers[key])
#     sorted_mixers = sorted(dict_mixers.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

#     return sorted_mixers

def find_mixer_gradients(dens_mat, mixer_dict, pauli_dict, graph, apply_ham_unitary=True, gamma_0=0.01, noisy=False, noise_prob=0.0):
    """
    Finds the absolute values of the gradients for all mixer operators.

    Parameters:
        dens_mat - DensityMatrix instance for which to find the energy gradients
        mixer_dict - Dictionary containing the mixer gradient operators for passed graph
        gamma_0 - parameter for cut hamiltonian unitary
        graph - Graph instance corresponding to problem at hand
    Returns:
        sorted_mixers - sorted list of tuples containing mixer type and absolute value of its
        gradient
    """

    if not isinstance(graph, nx.Graph):
        raise Exception("Error - passed graph must be instance of networkx Graph class!")

    if not isinstance(dens_mat, qi.DensityMatrix) and not isinstance(dens_mat, sparse._csr.csr_matrix):
        raise Exception("Error - passed density matrix is not an instance of a valid correct class!")

    if not isinstance(mixer_dict, dict):
        raise Exception("Error - passed mixers are not in a dictionary!")

    if apply_ham_unitary and gamma_0 != 0.0:

        cut_unit = build_operators.cut_unitary(graph, gamma_0, pauli_dict)
        new_dens_mat = (cut_unit * dens_mat) * (cut_unit.transpose().conj())
        if noisy:
            dens_mat = noisy_ham_unitary_evolution(dens_mat, noise_prob=noise_prob, graph=graph, pauli_dict=pauli_dict)

    else:

        new_dens_mat = dens_mat

    dict_gradients = {
        'standard_x' : 0.0,
        'standard_y' : 0.0
    }

    for mixer_type in mixer_dict:

        dict_gradients[mixer_type] = mixer_dict[mixer_type].find_exact_gradient(new_dens_mat)

        # check if mixer is a single-qubit Pauli
        no_qubits = 0
        for letter in mixer_type:
            if letter in ['X', 'Y', 'Z']:
                no_qubits += 1
            if no_qubits > 1:
                break

        if no_qubits == 1:

            if mixer_type[0] == 'X':
                dict_gradients['standard_x'] += dict_gradients[mixer_type]

            if mixer_type[0] == 'Y':
                dict_gradients['standard_y'] += dict_gradients[mixer_type]

        dict_gradients[mixer_type] = abs(dict_gradients[mixer_type])

    dict_gradients['standard_x'] = abs(dict_gradients['standard_x'])
    dict_gradients['standard_y'] = abs(dict_gradients['standard_y'])

    # sort gradients by absolute value and find best mixer
    sorted_gradients = sorted(dict_gradients.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    return sorted_gradients

# def find_mixer_gradients_new_algo(dens_mat, mixer_dict, graph):
#     """
#     Finds the absolute values of the gradients for all mixer operators.

#     Parameters:
#         dens_mat - DensityMatrix instance for which to find the energy gradients
#         mixer_dict - Dictionary containing the mixer gradient operators for passed graph
#         gamma_0 - parameter for cut hamiltonian unitary
#         graph - Graph instance corresponding to problem at hand
#     Returns:
#         sorted_mixers - sorted list of tuples containing mixer type and absolute value of its
#         gradient
#     """

#     if not isinstance(graph, Graph):
#         raise Exception("Error - passed graph must be instance of networkx Graph class!")

#     if not isinstance(dens_mat, qi.DensityMatrix):
#         raise Exception("Error - passed density matrix is not DensityMatrix instance!")

#     if not isinstance(mixer_dict, dict):
#         raise Exception("Error - passed mixers are not in a dictionary!")

#     if graph.number_of_nodes() != dens_mat.num_qubits:
#         raise Exception("Error - number of nodes in graph does not agree with number of qubits in quantum circuit!")

#     no_qubits = dens_mat.num_qubits
#     new_dens_mat_sparse = sparse.csr_matrix(dens_mat.data)

#     dict_gradients = {
#         'standard_x' : 0.0,
#         'standard_y' : 0.0
#     }

#     for mixer_type in mixer_dict:

#         dict_gradients[mixer_type] = mixer_dict[mixer_type].find_exact_gradient(new_dens_mat_sparse)

#         if len(mixer_type) == 2:

#             if mixer_type[0] == 'X':
#                 dict_gradients['standard_x'] += dict_gradients[mixer_type]

#             if mixer_type[0] == 'Y':
#                 dict_gradients['standard_y'] += dict_gradients[mixer_type]

#         dict_gradients[mixer_type] = abs(dict_gradients[mixer_type])

#     dict_gradients['standard_x'] = abs(dict_gradients['standard_x'])
#     dict_gradients['standard_y'] = abs(dict_gradients['standard_y'])

#     # sort gradients by absolute value and find best mixer
#     sorted_gradients = sorted(dict_gradients.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

#     return sorted_gradients


# def get_obj_func_adapt(graph, hamiltonian_operator, mixer_list, noise=False):
#     """
#     Returns objective function for minimisation procedure for
#     ADAPT-QAOA.
#     """
    
#     def execute_circ(parameter_values):
        
#         dens_mat = build_adapt_qaoa_ansatz(graph, parameter_values, mixer_list, noise=noise)
#         expectation_value = dens_mat.expectation_value(hamiltonian_operator)
#         return expectation_value.real * (-1.0)
    
#     return execute_circ

# def get_obj_func_adapt_new(graph, hamiltonian_operator, mixer_list, noise=False):
#     """
#     Returns objective function for minimisation procedure for
#     ADAPT-QAOA.
#     """
#     hamiltonian_sparse_matrix = sparse.csc_matrix(hamiltonian_operator.data)
#     def execute_circ(parameter_values):
        
#         dens_mat = build_adapt_qaoa_ansatz_new(graph, parameter_values, mixer_list, noise=noise)
#         expectation_value = (sparse.csc_matrix(dens_mat.data) * hamiltonian_sparse_matrix).trace()
#         return expectation_value.real * (-1.0)
    
#     return execute_circ

# def get_obj_func_standard(graph, hamiltonian_operator, noise=False):
#     """
#     Returns objective function for minimisation procedure for
#     Standard QAOA.
#     """
    
#     def execute_circ(parameter_values):
        
#         dens_mat = build_standard_qaoa_ansatz(graph, parameter_values, noise=noise)
#         expectation_value = dens_mat.expectation_value(hamiltonian_operator)
#         return expectation_value.real * (-1.0)
    
#     return execute_circ

def find_optimal_cut(graph):
    """
    Find optimal cut for passed graph using exhaustive search.

    Returns:
        solution - list containing:
            - max-cut represented by bitstring
            - max-cut value
            - max Hamiltonian eigenvalue
    """

    if not isinstance(graph, nx.Graph):
        raise Exception("Error - passed graph is not instance of Graph networkx class!")

    solution = []

    with warnings.catch_warnings():

        warnings.simplefilter("ignore", category=FutureWarning)
        offset = nx.adjacency_matrix(graph).sum() / 4
    
    for i in range(2**graph.number_of_nodes()):

        bitstring = bin(i)[2:]
        bitstring = '0' * (graph.number_of_nodes() - len(bitstring)) + bitstring
        cut = evaluate_cut(graph, bitstring)
        if len(solution) == 0:

            solution = [bitstring, cut, cut - offset]

        elif solution[1] < cut:

            solution = [bitstring, cut, cut - offset]

    return solution

def evaluate_cut(graph, bitstring):
    """
    Evaluate value of cut for passed graph.
    """

    if not isinstance(graph, nx.Graph):
        raise Exception("Error - passed graph is not instance of Graph networkx class!")

    cut = 0

    for edge in graph.edges:

        weight = graph.get_edge_data(*edge)['weight']
        if bitstring[int(edge[0])] != bitstring[int(edge[1])]:
            cut += weight

    return cut

def goemans_williamson(graph):
    """
    Returns the non-relaxed solution to the 
    SDP problem in the Goemans-WIlliamson algorithm
    for the passed graph.
    """

    if not isinstance(graph, nx.Graph):
        raise Exception("Error - passed graph is not an instance of the networkx Graph class!")

    dim = graph.number_of_nodes()

    # define semidefinite problem
    X = cvx.Variable((dim, dim), symmetric=True)
    constraints = [X >> 0] # semidefinite matrix
    constraints += [X[i,i] == 1 for i in range(dim)] # normalisation of relaxation

    objective = sum(0.5 * graph.get_edge_data(i, j)['weight'] * (1 - X[i, j]) for (i, j) in graph.edges)

    prob = cvx.Problem(cvx.Maximize(objective), constraints)

    prob.solve()

    x = linalg.sqrtm(X.value).real

    return x


def split_hamiltonian(graph, mixer):

    no_nodes = graph.number_of_nodes()
    all_pairs = [(i, j) for i in range(no_nodes) for j in range(i+1, no_nodes)]
    commuting_pairs = []
    anticommuting_pairs = []

    for pair in all_pairs:

        # check if mixer is a single-qubit Pauli
        no_qubits = 0
        for letter in mixer:
            if letter in ['X', 'Y', 'Z']:
                no_qubits += 1
            if no_qubits > 1:
                break

        if no_qubits == 1:

            if pair[0] == int(mixer[1]) or pair[1] == int(mixer[1]):
                anticommuting_pairs.append(pair)
            else:
                commuting_pairs.append(pair)

        elif no_qubits == 2:

            index = -1
            for letter in mixer:
                index += 1
                if index == 0:
                    pauli_1 = letter
                else:
                    if letter in ['X', 'Y', 'Z']:
                        qubit_1 = int(mixer[1:index])
                        pauli_2 = letter
                        qubit_2 = int(mixer[index+1:])
                        break
            if qubit_1 > qubit_2:
                pauli_1, pauli_2 = pauli_2, pauli_1
                qubit_1, qubit_2 = qubit_2, qubit_1

            if pair[0] == qubit_1 and pair[1] == qubit_2:
                if pauli_1 != 'Z' and pauli_2 != 'Z':
                    commuting_pairs.append(pair)
                else:
                    anticommuting_pairs.append(pair)
            elif pair[0] == qubit_1 and pair[1] != qubit_2:
                if pauli_1 == 'Z':
                    commuting_pairs.append(pair)
                else:
                    anticommuting_pairs.append(pair)
            elif pair[0] != qubit_1 and pair[1] == qubit_2:
                if pauli_2 == 'Z':
                    commuting_pairs.append(pair)
                else:
                    anticommuting_pairs.append(pair)
            elif pair[1] == qubit_1 and pair[0] != qubit_2:
                if pauli_1 == 'Z':
                    commuting_pairs.append(pair)
                else:
                    anticommuting_pairs.append(pair)
            elif pair[1] != qubit_1 and pair[0] == qubit_2:
                if pauli_2 == 'Z':
                    commuting_pairs.append(pair)
                else:
                    anticommuting_pairs.append(pair)
            else:
                commuting_pairs.append(pair)

        else:

            raise Exception('Error - Invalid Mixer Type!')

    # print('Commuting pairs\n', commuting_pairs)
    # print('Anticommuting pairs\n', anticommuting_pairs)
    
    # build commuting Hamiltonian
    if len(commuting_pairs) == 0:
        ham_commute = sparse.csr_matrix(dtype=complex, shape=(2**no_nodes, 2**no_nodes))
    else:
        pauli_strings = [None] * len(commuting_pairs)
        coeffs = [None] * len(commuting_pairs)
        for index, pair in enumerate(commuting_pairs):
            if graph.get_edge_data(*pair) != None:
                tmp_string = 'I' * (pair[0]) + 'Z' + 'I' * (pair[1]-pair[0]-1) + 'Z' + 'I' * (no_nodes - pair[1] - 1)
                tmp_string = tmp_string[::-1]
                pauli_strings[index] = tmp_string
                coeffs[index] = -0.5 * graph.get_edge_data(*pair)['weight']
        ham_commute = sparse.csr_matrix(qi.SparsePauliOp(pauli_strings, np.array(coeffs)).to_matrix())

    # build anticommuting Hamiltonian
    if len(anticommuting_pairs) == 0:
        ham_anticommute = sparse.csr_matrix((2**no_nodes, 2**no_nodes), dtype=complex)
    else:
        pauli_strings = [None] * len(anticommuting_pairs)
        coeffs = [None] * len(anticommuting_pairs)
        for index, pair in enumerate(anticommuting_pairs):
            if graph.get_edge_data(*pair) != None:
                tmp_string = 'I' * (pair[0]) + 'Z' + 'I' * (pair[1]-pair[0]-1) + 'Z' + 'I' * (no_nodes - pair[1] - 1)
                tmp_string = tmp_string[::-1]
                pauli_strings[index] = tmp_string
                coeffs[index] = -0.5 * graph.get_edge_data(*pair)['weight']
        ham_anticommute = sparse.csr_matrix(qi.SparsePauliOp(pauli_strings, np.array(coeffs)).to_matrix())

    return ham_commute, ham_anticommute


def calculate_mixer_expectations(dens_mat, mixer_type, ham_anticommute, mixer_paulis_dict, atol=1e-10):
    """
    Method which returns the expectation values
    which can be used to detemine whether the 
    Hamiltoinan unitary is required, or not.
    """

    if isinstance(ham_anticommute, dict):

        ham_anticommute = ham_anticommute[mixer_type]['H_a']

    # if not mixers_split_dict:
    #     ham_anticommute = 
    # split_for_curr_mixer = mixers_split_dict[mixer_type]
    # ham_anticommute = split_for_curr_mixer['H_a']
    mixer_operator = mixer_paulis_dict[mixer_type]

    result = {}
    MH_a = (dens_mat * (mixer_operator * ham_anticommute)).trace()
    iMH_a = (1j * MH_a).real
    if abs(iMH_a) < atol:
        iMH_a = 0
    MH_aH_a = (dens_mat * ((mixer_operator * ham_anticommute) * ham_anticommute)).trace().real
    if abs(MH_aH_a) < atol:
        MH_aH_a = 0
    MH_aH_aH_a = (dens_mat * (((mixer_operator * ham_anticommute) * ham_anticommute) * ham_anticommute)).trace()
    iMH_aH_aH_a = (1j * MH_aH_aH_a).real
    if abs(iMH_aH_aH_a) < atol:
        iMH_aH_aH_a = 0
    result['iMH_a'] = iMH_a
    result['MH_a^2'] = MH_aH_a
    result['iMH_a^3'] = iMH_aH_aH_a

    return result

####################
# noisy evolutions #
####################

# def apply_noise_channel(density_matrix, qubit, noise_prob):

#     if not isinstance(density_matrix, sparse._csr.csr_matrix):
#         raise Exception

#     new_density_matrix = (1-noise_prob) * density_matrix

#     no_qubits = round(math.log2(density_matrix.shape[0]))
#     paulis = {}
#     for pauli_type in ['X', 'Y', 'Z']:
#         pauli = QuantumCircuit(no_qubits)
#         if pauli_type == 'X':
#             pauli.x(qubit)
#         elif pauli_type == 'Y':
#             pauli.y(qubit)
#         elif pauli_type == 'Z':
#             pauli.z(qubit)
#         paulis[pauli_type] = sparse.csr_matrix(qi.Operator(pauli).data)
#         new_density_matrix = new_density_matrix + (noise_prob/3) * ((paulis[pauli_type] * density_matrix) * paulis[pauli_type])

    return new_density_matrix

def apply_noise_channel(density_matrix, qubit, noise_prob, pauli_dict):

    if not isinstance(density_matrix, sparse._csr.csr_matrix):
        raise Exception

    new_density_matrix = (1-noise_prob) * density_matrix

    for pauli_type in ['X', 'Y', 'Z']:
        
        new_density_matrix = new_density_matrix + (noise_prob/3) * ((pauli_dict[pauli_type + str(qubit)] * density_matrix) * pauli_dict[pauli_type + str(qubit)])

    return new_density_matrix

# def noisy_mixer_unitary_evolution(density_matrix, noise_prob, mixer_type):

#     if not isinstance(density_matrix, sparse._csr.csr_matrix):
#         raise Exception
    
#     new_dens_mat = density_matrix

#     if len(mixer_type) != 4:

#         return new_dens_mat
    
#     target_qubit = int(mixer_type[1])
#     new_dens_mat = apply_noise_channel(new_dens_mat, target_qubit, noise_prob)
#     new_dens_mat = apply_noise_channel(new_dens_mat, target_qubit, noise_prob)

#     return new_dens_mat

def noisy_mixer_unitary_evolution(density_matrix, noise_prob, mixer_type, pauli_dict):

    if not isinstance(density_matrix, sparse._csr.csr_matrix):
        raise Exception
    
    new_dens_mat = density_matrix

    # check if mixer is a single-qubit Pauli
    no_qubits = 0
    for letter in mixer_type:
        if letter in ['X', 'Y', 'Z']:
            no_qubits += 1
        if no_qubits > 1:
            break
    
    if no_qubits != 2:

        return new_dens_mat

    index = -1
    for letter in mixer_type:
        index += 1
        if index == 0:
            continue
        else:
            if letter in ['X', 'Y', 'Z']:
                target_qubit = int(mixer_type[1:index])
                break

    new_dens_mat = apply_noise_channel(new_dens_mat, target_qubit, noise_prob, pauli_dict=pauli_dict)
    new_dens_mat = apply_noise_channel(new_dens_mat, target_qubit, noise_prob, pauli_dict=pauli_dict)

    return new_dens_mat
    
# def noisy_ham_unitary_evolution(density_matrix, noise_prob, graph):

#     if not isinstance(density_matrix, sparse._csr.csr_matrix):
#         raise Exception
    
#     if not isinstance(graph, nx.Graph):
#         raise Exception

#     no_qubits = graph.number_of_nodes()

#     new_dens_mat = density_matrix

#     for pair in list(graph.edges()):

#         new_dens_mat = apply_noise_channel(new_dens_mat, pair[1], noise_prob)
#         new_dens_mat = apply_noise_channel(new_dens_mat, pair[1], noise_prob)

#     return new_dens_mat

def noisy_ham_unitary_evolution(density_matrix, noise_prob, graph, pauli_dict):

    if not isinstance(density_matrix, sparse._csr.csr_matrix):
        raise Exception
    
    if not isinstance(graph, nx.Graph):
        raise Exception

    new_dens_mat = density_matrix

    for pair in list(graph.edges()):

        new_dens_mat = apply_noise_channel(new_dens_mat, pair[1], noise_prob, pauli_dict=pauli_dict)
        new_dens_mat = apply_noise_channel(new_dens_mat, pair[1], noise_prob, pauli_dict=pauli_dict)

    return new_dens_mat
