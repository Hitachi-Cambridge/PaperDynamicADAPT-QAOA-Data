"""
Python file with methods for running QAOA algorithms and recording data.
"""

# imports
from networkx import Graph
from src_code import build_operators
from src_code import useful_methods
from scipy.optimize import minimize
import numpy as np

######################
# quantum algorithms #
######################

def run_adapt_qaoa(graph, pauli_ops_dict, gradient_ops_dict, max_depth, beta_0=0.0, gamma_0 = 0.01, rel_gtol = 10**-2, etol=-1):
    """
    Method which runs ADAPT-QAOA.

    Parameters:
        graph - Networkx Graph instance of graph on which to run algorithm
        pauli_ops_dict - dictionary of Pauli sparse matrices for unitary generation
        gradient_ops_dict - dictionary of mixers for which gradients will be evaluated
        max_depth - maximum depth of ansatz
        beta_0 - initial parameter guess for mixer unitaries
        gamma_0 - initial parameter guess for Hamiltonian unitaries
        rel_gtol - factor by which to multiply current maximum gradient when
            setting the gtol for the classical optimiser
        etol - tolerance for absolute energy expectation improvement (default = -1, i.e., this criterion is not considered)

    Returns:
        results:
            - achieved max-cut approximation ratios at each layer
            - achieved Hamiltonian ground state approximation ratios at each layer
            - optimal mixers at each layer
            - optimal mixer unitary parameters
            - optimal Hamiltonian unitary parameters
            - dictionary of gradient magnitudes for each mixer at each layer
    """

    if not isinstance(graph, Graph):
        raise Exception("Error - passed graph is not instance of Networkx Graph class!")

    max_cut_solution = useful_methods.find_optimal_cut(graph)
    max_cut_value = max_cut_solution[1]
    max_ham_value = max_cut_solution[2]
    ham_offset = max_cut_value - max_ham_value

    hamiltonian = build_operators.cut_hamiltonian(graph)

    mixer_params = []
    mixer_list = []
    ham_params = []
    ham_layers = []
    ham_approx_ratios = []
    cut_approx_ratios = []
    all_mixers_per_layer_dict = {}

    curr_layer = 0
    curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers)
    curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
    cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
    print('Initial Cut Approximation Ratio:', cut_approx_ratio, '\n')
    ham_approx_ratios.append(curr_ham_estimate)
    cut_approx_ratios.append(cut_approx_ratio)

    def obj_func(parameter_values, mixers, ham_unitary_layers):
        
        no_ham_layers = len(ham_unitary_layers)
        no_params = len(parameter_values)
        dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, parameter_values[:(no_params-no_ham_layers)], mixers, parameter_values[(no_params-no_ham_layers):], pauli_ops_dict, ham_unitary_layers)
        expectation_value = (hamiltonian * dens_mat).trace().real
        return expectation_value * (-1.0)

    not_converged = True
    while not_converged:

        curr_layer += 1
        print("Finding Best Mixer for layer " + str(curr_layer) + "...")
        all_mixer_gradients = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=True)
        

        mixer_list.append(all_mixer_gradients[0][0])
        gradient_tolerance = all_mixer_gradients[0][1] * rel_gtol
        print("\tBest mixer is " + str(all_mixer_gradients[0][0]) + " with gradient magnitude " + str(all_mixer_gradients[0][1]) + '\n')

        if not all_mixers_per_layer_dict:

            all_mixers_per_layer_dict = {x[0]:[x[1]] for x in all_mixer_gradients}

        else:

            for mixer in all_mixer_gradients:

                all_mixers_per_layer_dict[mixer[0]].append(mixer[1])

        print("Optimising layer " + str(curr_layer) + "...")
        initial_parameter_guesses = mixer_params + [beta_0] + ham_params + [gamma_0]
        ham_layers.append(curr_layer)
        print('\tInitial Parameter Guesses:', initial_parameter_guesses)
        optimiser_options = {
            'gtol' : gradient_tolerance
        }
        result = minimize(obj_func, initial_parameter_guesses, method="BFGS", args=(mixer_list, ham_layers), options=optimiser_options)

        print('\tOptimisation completed wih following outcome:')
        print('\t\tNumber of iterations performed: ' + str(result.nit))
        print('\t\tNumber of expectation evaluations performed: ' + str(result.nfev))
        print('\t\tSuccess: ' + str(result.success))
        print('\t\tOptimiser message: ' + str(result.message))

        parameter_list = list(result.x)
        mixer_params = parameter_list[:curr_layer]
        ham_params = parameter_list[curr_layer:]
        mixer_params_string = '\tOptimised mixer unitary parameters: '
        for param in mixer_params:
            mixer_params_string += f'{param:.3}' + ', '
        ham_params_string = '\tOptimised Hamiltonian unitary parameters: '
        for param in ham_params:
            ham_params_string += f'{param:.3}' + ', '

        print(mixer_params_string[:-2])
        print(ham_params_string[:-2])
        curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers)
        curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
        ham_approx_ratios.append(curr_ham_estimate)
        cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
        cut_approx_ratios.append(cut_approx_ratio)
        print('\nCurrent Cut Approximation Ratio:', cut_approx_ratio)

        # check convergence
        if curr_layer == max_depth or abs(ham_approx_ratios[-1]-ham_approx_ratios[-2]) < etol:
            not_converged = False

        print('\n')

    for layer in range(len(ham_approx_ratios)):
        ham_approx_ratios[layer] /= max_ham_value

    data = {
        'cut_approx_ratios' : cut_approx_ratios,
        'ham_approx_ratios' : ham_approx_ratios,
        'best_mixers' : mixer_list,
        'best_mixer_parameters' : mixer_params,
        'best_ham_parameters' : ham_params,
        'all_mixers' : all_mixers_per_layer_dict
    }

    return data

def run_dynamic_adapt_qaoa(graph, pauli_ops_dict, gradient_ops_dict, pauli_mixers_split_ops_dict, max_depth, beta_0=0.0, gamma_tilde = 0.1, rel_gtol = 10**-2, delta_1 = 0.0, delta_2=1e-8, etol=-1):
    """
    Method which runs Dynamic ADAPT-QAOA.

    Parameters:
        graph - Networkx Graph instance of graph on which to run algorithm
        pauli_ops_dict - dictionary of Pauli sparse matrices for unitary generation
        gradient_ops_dict - dictionary of mixers for which gradients will be evaluated
        pauli_mixers_split_ops_dict - dictionary of Pauli sparse matrices for unitary generation split into
            commuting and anti-commuting parts w.r.t. the Hamiltonian
        max_depth - maximum depth of ansatz
        beta_0 - initial parameter guess for mixer unitaries
        gamma_tilde - parameter guess for Hamiltonian unitaries if kept in layer
        rel_gtol - factor by which to multiply current maximum gradient when
            setting the gtol for the classical optimiser
        delta_1, delta_2 - error tolerances for Hamiltonian unitary removal check
        etol - tolerance for absolute energy expectation improvement (default = -1, i.e., this criterion is not considered)

    Returns:
        results:
            - achieved max-cut approximation ratios at each layer
            - achieved Hamiltonian ground state approximation ratios at each layer
            - optimal mixers at each layer
            - optimal mixer unitary parameters
            - optimal Hamiltonian unitary parameters
            - layer numbers at which cost unitaties are included
            - dictionary of gradient magnitudes for each mixer at each layer
    """

    if not isinstance(graph, Graph):
        raise Exception("Error - passed graph is not instance of Networkx Graph class!")

    max_cut_solution = useful_methods.find_optimal_cut(graph)
    max_cut_value = max_cut_solution[1]
    max_ham_value = max_cut_solution[2]
    ham_offset = max_cut_value - max_ham_value

    hamiltonian = build_operators.cut_hamiltonian(graph)

    mixer_params = []
    mixer_list = []
    ham_params = []
    ham_layers = []
    ham_approx_ratios = []
    cut_approx_ratios = []
    all_mixers_per_layer_dict = {}

    curr_layer = 0
    curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers)
    curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
    cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
    print('Initial Cut Approximation Ratio:', cut_approx_ratio, '\n')
    ham_approx_ratios.append(curr_ham_estimate / max_ham_value)
    cut_approx_ratios.append(cut_approx_ratio)

    def obj_func(parameter_values, mixers, ham_unitary_layers):
        
        no_ham_layers = len(ham_unitary_layers)
        no_params = len(parameter_values)
        dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, parameter_values[:(no_params-no_ham_layers)], mixers, parameter_values[(no_params-no_ham_layers):], pauli_ops_dict, ham_unitary_layers)
        expectation_value = (hamiltonian * dens_mat).trace().real
        return expectation_value * (-1.0)

    prev_best_mixer = None # cannot append the same mixer consecutive times in this ansatz if there is no Hamiltonian unitary in the current layer

    not_converged = True
    while not_converged:

        curr_layer += 1

        print("Finding Best Mixer for layer " + str(curr_layer) + "...")
        all_mixer_gradients = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=False)

        best_mixer = all_mixer_gradients[0][0]
        best_mixer_gradient = all_mixer_gradients[0][1]
        if curr_layer > 1 and prev_best_mixer == best_mixer:

            best_mixer = all_mixer_gradients[1][0]
            best_mixer_gradient = all_mixer_gradients[1][1]

        print('\tThe best mixer for layer ' + str(curr_layer) + ' with no Hamiltonian unitary is ' + best_mixer + ' with a gradient of ' + str(best_mixer_gradient))

        if 'standard' not in best_mixer: # can only perform analysis below for single Pauli string mixers

            # determine whether having not Hamiltonian unitary would correspond to a local maximum
            expectations = useful_methods.calculate_mixer_expectations(curr_dens_mat, best_mixer, pauli_mixers_split_ops_dict, pauli_ops_dict, atol=1e-8)
            print('\t<iMH_a> =', expectations['iMH_a'])
            print('\t<MH_a^2> =', expectations['MH_a^2'])
            print('\t<iMH_a^3> =', expectations['iMH_a^3'])
            hessian = 16 * expectations['iMH_a^3'] * expectations['iMH_a']

            if abs(expectations['MH_a^2']) <= delta_1 and hessian - delta_2 > 0:

                use_ham_unitary = False
                print('\tA maximum occurs when using no Hamiltonian unitary so we remove it for this layer!')
            
            else:

                use_ham_unitary = True
                print('\tIt is unclear whether a maximum occurs when using no Hamiltonian unitary so we will add it to the layer!')
                all_mixer_gradients_positive = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=use_ham_unitary, gamma_0=gamma_tilde)
                all_mixer_gradients_negative = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=use_ham_unitary, gamma_0=-1 * gamma_tilde)
                if all_mixer_gradients_negative[0][1] > all_mixer_gradients_positive[0][1]:
                    all_mixer_gradients = all_mixer_gradients_negative
                    gamma_guess = -1.0 * gamma_tilde
                else:
                    all_mixer_gradients = all_mixer_gradients_positive
                    gamma_guess = gamma_tilde
                best_mixer = all_mixer_gradients[0][0]
                best_mixer_gradient = all_mixer_gradients[0][1]
                print('\tThe new best mixer for layer ' + str(curr_layer) + ' with a Hamiltonian unitary is ' + best_mixer + ' with a gradient of ' + str(best_mixer_gradient))
        
        mixer_list.append(best_mixer)
        prev_best_mixer = best_mixer

        if not all_mixers_per_layer_dict:

            all_mixers_per_layer_dict = {x[0]:[x[1]] for x in all_mixer_gradients}

        else:

            for mixer in all_mixer_gradients:

                all_mixers_per_layer_dict[mixer[0]].append(mixer[1])

        print("\nOptimising layer " + str(curr_layer) + "...")
        if not use_ham_unitary:
            initial_parameter_guesses = mixer_params + [beta_0] + ham_params
        else:
            initial_parameter_guesses = mixer_params + [beta_0] + ham_params + [gamma_guess]
            ham_layers.append(curr_layer)
        print('\tInitial Parameter Guesses:', initial_parameter_guesses)
        optimiser_options = {
            'gtol' : best_mixer_gradient * rel_gtol,
        }
        result = minimize(obj_func, initial_parameter_guesses, method='BFGS', args=(mixer_list, ham_layers), options=optimiser_options)

        print('\tOptimisation completed wih following outcome:')
        print('\t\tNumber of iterations performed: ' + str(result.nit))
        print('\t\tNumber of expectation evaluations performed: ' + str(result.nfev))
        print('\t\tSuccess: ' + str(result.success))
        print('\t\tOptimiser message: ' + str(result.message))

        parameter_list = list(result.x)

        mixer_params = parameter_list[:curr_layer]
        ham_params = parameter_list[curr_layer:]
        mixer_params_string = '\tOptimised mixer unitary parameters: '
        for param in mixer_params:
            mixer_params_string += f'{param:.3}' + ', '
        ham_params_string = '\tOptimised Hamiltonian unitary parameters: '
        for param in ham_params:
            ham_params_string += f'{param:.3}' + ', '

        print(mixer_params_string[:-2])
        print(ham_params_string[:-2])

        curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers)
        curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
        ham_approx_ratios.append(curr_ham_estimate / max_ham_value)
        cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
        cut_approx_ratios.append(cut_approx_ratio)
        print('\nCurrent Cut Approximation Ratio:', cut_approx_ratio)

        # check convergence
        if curr_layer == max_depth or abs(ham_approx_ratios[-1]-ham_approx_ratios[-2]) < etol:
            not_converged = False

        print('\n')


    data = {
        'cut_approx_ratios' : cut_approx_ratios,
        'ham_approx_ratios' : ham_approx_ratios,
        'best_mixers' : mixer_list,
        'best_mixer_parameters' : mixer_params,
        'best_ham_parameters' : ham_params,
        'ham_unitary_layers' : ham_layers,
        'all_mixers' : all_mixers_per_layer_dict
    }

    return data

def run_adapt_qaoa_no_cost_unitaries(graph, pauli_ops_dict, gradient_ops_dict, max_depth, beta_0=0.0, rel_gtol = 10**-2, etol=-1):
    """
    Method which runs ADAPT-QAOA with no cost unitaries.

    Parameters:
        graph - Networkx Graph instance of graph on which to run algorithm
        pauli_ops_dict - dictionary of Pauli sparse matrices for unitary generation
        gradient_ops_dict - dictionary of mixers for which gradients will be evaluated
        max_depth - maximum depth of ansatz
        beta_0 - initial parameter guess for mixer unitaries
        rel_gtol - factor by which to multiply current maximum gradient when
            setting the gtol for the classical optimiser
        etol - tolerance for absolute energy expectation improvement (default = -1, i.e., this criterion is not considered)

    Returns:
        results:
            - achieved max-cut approximation ratios at each layer
            - achieved Hamiltonian ground state approximation ratios at each layer
            - optimal mixers at each layer
            - optimal mixer unitary parameters
            - dictionary of gradient magnitudes for each mixer at each layer
    """

    if not isinstance(graph, Graph):
        raise Exception("Error - passed graph is not instance of Networkx Graph class!")

    max_cut_solution = useful_methods.find_optimal_cut(graph)
    max_cut_value = max_cut_solution[1]
    max_ham_value = max_cut_solution[2]
    ham_offset = max_cut_value - max_ham_value

    hamiltonian = build_operators.cut_hamiltonian(graph)

    mixer_params = []
    mixer_list = []
    ham_params = []
    ham_layers = []
    ham_approx_ratios = []
    cut_approx_ratios = []
    all_mixers_per_layer_dict = {}

    curr_layer = 0
    curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers)
    curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
    cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
    print('Initial Cut Approximation Ratio:', cut_approx_ratio, '\n')
    ham_approx_ratios.append(curr_ham_estimate / max_ham_value)
    cut_approx_ratios.append(cut_approx_ratio)

    def obj_func(parameter_values, mixers, ham_unitary_layers):
        
        no_ham_layers = len(ham_unitary_layers)
        no_params = len(parameter_values)
        dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, parameter_values[:(no_params-no_ham_layers)], mixers, parameter_values[(no_params-no_ham_layers):], pauli_ops_dict, ham_unitary_layers)
        expectation_value = (hamiltonian * dens_mat).trace().real
        return expectation_value * (-1.0)

    prev_best_mixer = None # cannot append the same mixer consecutive times in this ansatz if there is no Hamiltonian unitary in the current layer

    not_converged = True
    while not_converged:

        curr_layer += 1

        print("Finding Best Mixer for layer " + str(curr_layer) + "...")
        all_mixer_gradients = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=False)

        best_mixer = all_mixer_gradients[0][0]
        best_mixer_gradient = all_mixer_gradients[0][1]
        if curr_layer > 1 and prev_best_mixer == best_mixer:

            best_mixer = all_mixer_gradients[1][0]
            best_mixer_gradient = all_mixer_gradients[1][1]

        print('\tThe best mixer for layer ' + str(curr_layer) + ' with no Hamiltonian unitary is ' + best_mixer + ' with a gradient of ' + str(best_mixer_gradient))

        mixer_list.append(best_mixer)
        prev_best_mixer = best_mixer

        if not all_mixers_per_layer_dict:

            all_mixers_per_layer_dict = {x[0]:[x[1]] for x in all_mixer_gradients}

        else:

            for mixer in all_mixer_gradients:

                all_mixers_per_layer_dict[mixer[0]].append(mixer[1])

        print("\nOptimising layer " + str(curr_layer) + "...")
        initial_parameter_guesses = mixer_params + [beta_0] + ham_params
        print('\tInitial Parameter Guesses:', initial_parameter_guesses)
        optimiser_options = {
            'gtol' : best_mixer_gradient * rel_gtol,
        }
        result = minimize(obj_func, initial_parameter_guesses, method='BFGS', args=(mixer_list, ham_layers), options=optimiser_options)

        print('\tOptimisation completed wih following outcome:')
        print('\t\tNumber of iterations performed: ' + str(result.nit))
        print('\t\tNumber of expectation evaluations performed: ' + str(result.nfev))
        print('\t\tSuccess: ' + str(result.success))
        print('\t\tOptimiser message: ' + str(result.message))

        parameter_list = list(result.x)

        mixer_params = parameter_list[:curr_layer]
        ham_params = parameter_list[curr_layer:]
        mixer_params_string = '\tOptimised mixer unitary parameters: '
        for param in mixer_params:
            mixer_params_string += f'{param:.3}' + ', '
        ham_params_string = '\tOptimised Hamiltonian unitary parameters: '
        for param in ham_params:
            ham_params_string += f'{param:.3}' + ', '

        print(mixer_params_string[:-2])
        # print(ham_params_string[:-2])

        curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers)
        curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
        ham_approx_ratios.append(curr_ham_estimate / max_ham_value)
        cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
        cut_approx_ratios.append(cut_approx_ratio)
        print('\nCurrent Cut Approximation Ratio:', cut_approx_ratio)

        # check convergence
        if curr_layer == max_depth or abs(ham_approx_ratios[-1]-ham_approx_ratios[-2]) < etol:
            not_converged = False

        print('\n')


    data = {
        'cut_approx_ratios' : cut_approx_ratios,
        'ham_approx_ratios' : ham_approx_ratios,
        'best_mixers' : mixer_list,
        'best_mixer_parameters' : mixer_params,
        'all_mixers' : all_mixers_per_layer_dict
    }

    return data

def run_dynamic_adapt_qaoa_noisy(graph, pauli_ops_dict, gradient_ops_dict, pauli_mixers_split_ops_dict, max_depth, beta_0=0.0, gamma_tilde = 0.1, rel_gtol = 10**-2, delta_1 = 0.0, delta_2=1e-8, gate_error_prob = 0.0, etol=-1):
    """
    Method which runs Dynamic ADAPT-QAOA in the presence of depolarizing two-qubit noise.

    Parameters:
        graph - Networkx Graph instance of graph on which to run algorithm
        pauli_ops_dict - dictionary of Pauli sparse matrices for unitary generation
        gradient_ops_dict - dictionary of mixers for which gradients will be evaluated
        pauli_mixers_split_ops_dict - dictionary of Pauli sparse matrices for unitary generation split into
            commuting and anti-commuting parts w.r.t. the Hamiltonian
        max_depth - maximum depth of ansatz
        beta_0 - initial parameter guess for mixer unitaries
        gamma_tilde - initial parameter guess for cost unitaries if kept in layer
        rel_gtol - factor by which to multiply current maximum gradient when
            setting the gtol for the classical optimiser
        delta_1, delta_2 - error tolerances for Hamiltonian unitary removal check
        gate_error_prob - probability of depolarizing noise error occuring in each CNOT gate
        etol - tolerance for absolute energy expectation improvement (default = -1, i.e., this criterion is not considered)

    Returns:
        results:
            - achieved max-cut approximation ratios at each layer
            - achieved Hamiltonian ground state approximation ratios at each layer
            - optimal mixers at each layer
            - optimal mixer unitary parameters
            - optimal Hamiltonian unitary parameters
            - layer numbers at which cost unitaties are included
            - dictionary of gradient magnitudes for each mixer at each layer
    """

    if not isinstance(graph, Graph):
        raise Exception("Error - passed graph is not instance of Networkx Graph class!")

    max_cut_solution = useful_methods.find_optimal_cut(graph)
    max_cut_value = max_cut_solution[1]
    max_ham_value = max_cut_solution[2]
    ham_offset = max_cut_value - max_ham_value

    hamiltonian = build_operators.cut_hamiltonian(graph)

    mixer_params = []
    mixer_list = []
    ham_params = []
    ham_layers = []
    ham_approx_ratios = []
    cut_approx_ratios = []
    all_mixers_per_layer_dict = {}

    curr_layer = 0
    curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers, noisy=True, noise_prob=gate_error_prob)
    curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
    cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
    print('Initial Cut Approximation Ratio:', cut_approx_ratio, '\n')
    ham_approx_ratios.append(curr_ham_estimate / max_ham_value)
    cut_approx_ratios.append(cut_approx_ratio)

    def obj_func(parameter_values, mixers, ham_unitary_layers):
        
        no_ham_layers = len(ham_unitary_layers)
        no_params = len(parameter_values)
        dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, parameter_values[:(no_params-no_ham_layers)], mixers, parameter_values[(no_params-no_ham_layers):], pauli_ops_dict, ham_unitary_layers, noisy=True, noise_prob=gate_error_prob)
        expectation_value = (hamiltonian * dens_mat).trace().real
        return expectation_value * (-1.0)

    prev_best_mixer = None # cannot append the same mixer consecutive times in this ansatz if there is no Hamiltonian unitary in the current layer

    not_converged = True
    while not_converged:

        curr_layer += 1

        print("Finding Best Mixer for layer " + str(curr_layer) + "...")
        all_mixer_gradients = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=False, noisy=True, noise_prob=gate_error_prob)

        best_mixer = all_mixer_gradients[0][0]
        best_mixer_gradient = all_mixer_gradients[0][1]
        if curr_layer > 1 and prev_best_mixer == best_mixer:

            best_mixer = all_mixer_gradients[1][0]
            best_mixer_gradient = all_mixer_gradients[1][1]

        print('\tThe best mixer for layer ' + str(curr_layer) + ' with no Hamiltonian unitary is ' + best_mixer + ' with a gradient of ' + str(best_mixer_gradient))

        if 'standard' not in best_mixer: # can only perform analysis below for single Pauli string mixers

            # determine whether having not Hamiltonian unitary would correspond to a local maximum
            expectations = useful_methods.calculate_mixer_expectations(curr_dens_mat, best_mixer, pauli_mixers_split_ops_dict, pauli_ops_dict, atol=1e-8)
            print('\t<iMH_a> =', expectations['iMH_a'])
            print('\t<MH_a^2> =', expectations['MH_a^2'])
            print('\t<iMH_a^3> =', expectations['iMH_a^3'])
            hessian = 16 * expectations['iMH_a^3'] * expectations['iMH_a']

            if abs(expectations['MH_a^2']) <= delta_1 and hessian - delta_2 > 0:

                use_ham_unitary = False
                print('\tA maximum occurs when using no Hamiltonian unitary so we remove it for this layer!')
            
            else:

                use_ham_unitary = True
                print('\tIt is unclear whether a maximum occurs when using no Hamiltonian unitary so we will add it to the layer!')
                all_mixer_gradients_positive = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=use_ham_unitary, gamma_0=gamma_tilde, noisy=True, noise_prob=gate_error_prob)
                all_mixer_gradients_negative = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=use_ham_unitary, gamma_0=-1 * gamma_tilde, noisy=True, noise_prob=gate_error_prob)
                if all_mixer_gradients_negative[0][1] > all_mixer_gradients_positive[0][1]:
                    all_mixer_gradients = all_mixer_gradients_negative
                    gamma_guess = -1.0 * gamma_tilde
                else:
                    all_mixer_gradients = all_mixer_gradients_positive
                    gamma_guess = gamma_tilde
                best_mixer = all_mixer_gradients[0][0]
                best_mixer_gradient = all_mixer_gradients[0][1]
                print('\tThe new best mixer for layer ' + str(curr_layer) + ' with a Hamiltonian unitary is ' + best_mixer + ' with a gradient of ' + str(best_mixer_gradient))
        
        mixer_list.append(best_mixer)
        prev_best_mixer = best_mixer

        if not all_mixers_per_layer_dict:

            all_mixers_per_layer_dict = {x[0]:[x[1]] for x in all_mixer_gradients}

        else:

            for mixer in all_mixer_gradients:

                all_mixers_per_layer_dict[mixer[0]].append(mixer[1])

        print("\nOptimising layer " + str(curr_layer) + "...")
        if not use_ham_unitary:
            initial_parameter_guesses = mixer_params + [beta_0] + ham_params
        else:
            initial_parameter_guesses = mixer_params + [beta_0] + ham_params + [gamma_guess]
            ham_layers.append(curr_layer)
        print('\tInitial Parameter Guesses:', initial_parameter_guesses)
        optimiser_options = {
            'gtol' : best_mixer_gradient * rel_gtol,
        }
        result = minimize(obj_func, initial_parameter_guesses, method='BFGS', args=(mixer_list, ham_layers), options=optimiser_options)

        print('\tOptimisation completed wih following outcome:')
        print('\t\tNumber of iterations performed: ' + str(result.nit))
        print('\t\tNumber of expectation evaluations performed: ' + str(result.nfev))
        print('\t\tSuccess: ' + str(result.success))
        print('\t\tOptimiser message: ' + str(result.message))

        parameter_list = list(result.x)

        mixer_params = parameter_list[:curr_layer]
        ham_params = parameter_list[curr_layer:]
        mixer_params_string = '\tOptimised mixer unitary parameters: '
        for param in mixer_params:
            mixer_params_string += f'{param:.3}' + ', '
        ham_params_string = '\tOptimised Hamiltonian unitary parameters: '
        for param in ham_params:
            ham_params_string += f'{param:.3}' + ', '

        print(mixer_params_string[:-2])
        print(ham_params_string[:-2])

        curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers, noisy=True, noise_prob=gate_error_prob)
        curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
        ham_approx_ratios.append(curr_ham_estimate / max_ham_value)
        cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
        cut_approx_ratios.append(cut_approx_ratio)
        print('\nCurrent Cut Approximation Ratio:', cut_approx_ratio)

        # check convergence
        if curr_layer == max_depth or abs(ham_approx_ratios[-1]-ham_approx_ratios[-2]) < etol:
            not_converged = False

        print('\n')


    data = {
        'cut_approx_ratios' : cut_approx_ratios,
        'ham_approx_ratios' : ham_approx_ratios,
        'best_mixers' : mixer_list,
        'best_mixer_parameters' : mixer_params,
        'best_ham_parameters' : ham_params,
        'ham_unitary_layers' : ham_layers,
        'all_mixers' : all_mixers_per_layer_dict
    }

    return data

def run_standard_qaoa(graph, depth, pauli_ops_dict, gamma_0=0.01, beta_0 = 0.0):
    """
    Method which runs the standard QAOA algorithm using density matrix
    formalism.

    Parameters:
        graph - Networkx Graph instance of graph on which to run algorithm.
        depth - depth of Ansatz
        pauli_ops_dict - dictionary of Pauli sparse matrices for unitary generation
        beta_0 - initial parameter guess for mixer unitaries
        gamma_0 - initial parameter guess for cost unitaries

    Returns:
        results:
            - achieved max-cut approximation ratio
            - achieved Hamiltonian max eigenvalue approximation ratio
            - optimal Hamiltonian unitary parameters
            - optimal mixer unitary parameters
    """

    if not isinstance(graph, Graph):
        raise Exception("Error - passed graph is not instance of Networkx Graph class!")

    max_cut_solution = useful_methods.find_optimal_cut(graph)
    max_cut_value = max_cut_solution[1]
    max_ham_eigenvalue = max_cut_solution[2]

    hamiltonian = build_operators.cut_hamiltonian(graph)
    hamiltonian_expectation = None

    def obj_func(parameter_values):
        
        dens_mat = build_operators.build_standard_qaoa_ansatz(graph, parameter_values, pauli_ops_dict)
        expectation_value = (hamiltonian * dens_mat).trace().real
        return expectation_value * (-1.0)

    initial_parameter_guesses = [gamma_0] * (depth) + [beta_0] * (depth)
    result = minimize(obj_func, initial_parameter_guesses, method="BFGS")

    parameter_list = list(result.x)

    dens_mat = build_operators.build_standard_qaoa_ansatz(graph, parameter_list, pauli_ops_dict)
    hamiltonian_expectation = (hamiltonian * dens_mat).trace().real
    ham_approx_ratio = hamiltonian_expectation / max_ham_eigenvalue
    cut_approx_ratio = (hamiltonian_expectation + max_cut_value - max_ham_eigenvalue) / max_cut_value

    data = {
        'cut_approx_ratio' : cut_approx_ratio,
        'ham_approx_ratio' : ham_approx_ratio,
        'optimised_Hamiltonian_unitary_parameters' : parameter_list[:depth],
        'optimised_mixer_unitary_parameters' : parameter_list[depth:],
    }

    return data

########################################
# quantum algorithms (silent versions) #
##### suitable for parallelisation #####
########################################

def run_adapt_qaoa_silent(graph, pauli_ops_dict, gradient_ops_dict, max_depth, beta_0=0.0, gamma_0 = 0.01, rel_gtol = 10**-2, output_file = 'adapt_qaoa.txt', etol=-1):
    """
    Method which runs ADAPT-QAOA.

    Parameters:
        graph - Networkx Graph instance of graph on which to run algorithm
        pauli_ops_dict - dictionary of Pauli sparse matrices for unitary generation
        gradient_ops_dict - dictionary of mixers for which gradients will be evaluated
        max_depth - maximum depth of ansatz
        beta_0 - initial parameter guess for mixer unitaries
        gamma_0 - initial parameter guess for Hamiltonian unitaries
        rel_gtol - factor by which to multiply current maximum gradient when
            setting the gtol for the classical optimiser
        output_file - .txt file into which to write simulator output
        etol - tolerance for absolute energy expectation improvement (default = -1, i.e., this criterion is not considered)

    Returns:
        results:
            - achieved max-cut approximation ratios at each layer
            - achieved Hamiltonian ground state approximation ratios at each layer
            - optimal mixers at each layer
            - optimal mixer unitary parameters
            - optimal Hamiltonian unitary parameters
            - dictionary of gradient magnitudes for each mixer at each layer
    """

    output_writer = open(output_file, 'w')

    if not isinstance(graph, Graph):
        raise Exception("Error - passed graph is not instance of Networkx Graph class!")

    max_cut_solution = useful_methods.find_optimal_cut(graph)
    max_cut_value = max_cut_solution[1]
    max_ham_value = max_cut_solution[2]
    ham_offset = max_cut_value - max_ham_value

    hamiltonian = build_operators.cut_hamiltonian(graph)

    mixer_params = []
    mixer_list = []
    ham_params = []
    ham_layers = []
    ham_approx_ratios = []
    cut_approx_ratios = []
    all_mixers_per_layer_dict = {}

    curr_layer = 0
    curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers)
    curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
    cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
    output_writer.write('Initial Cut Approximation Ratio: ' + str(cut_approx_ratio) + '\n\n')
    ham_approx_ratios.append(curr_ham_estimate)
    cut_approx_ratios.append(cut_approx_ratio)

    def obj_func(parameter_values, mixers, ham_unitary_layers):
        
        no_ham_layers = len(ham_unitary_layers)
        no_params = len(parameter_values)
        dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, parameter_values[:(no_params-no_ham_layers)], mixers, parameter_values[(no_params-no_ham_layers):], pauli_ops_dict, ham_unitary_layers)
        expectation_value = (hamiltonian * dens_mat).trace().real
        return expectation_value * (-1.0)

    not_converged = True
    while not_converged:

        curr_layer += 1
        output_writer.write("Finding Best Mixer for layer " + str(curr_layer) + "...\n")
        all_mixer_gradients = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=True)
        

        mixer_list.append(all_mixer_gradients[0][0])
        gradient_tolerance = all_mixer_gradients[0][1] * rel_gtol
        output_writer.write("\tBest mixer is " + str(all_mixer_gradients[0][0]) + " with gradient magnitude " + str(all_mixer_gradients[0][1]) + '\n\n')

        if not all_mixers_per_layer_dict:

            all_mixers_per_layer_dict = {x[0]:[x[1]] for x in all_mixer_gradients}

        else:

            for mixer in all_mixer_gradients:

                all_mixers_per_layer_dict[mixer[0]].append(mixer[1])

        output_writer.write("Optimising layer " + str(curr_layer) + "...\n")
        initial_parameter_guesses = mixer_params + [beta_0] + ham_params + [gamma_0]
        ham_layers.append(curr_layer)
        initial_parameters_string = '\tInitial parameter guesses: '
        for param in initial_parameter_guesses:
            initial_parameters_string += f'{param:.3}' + ', '
        output_writer.write(initial_parameters_string[:-2] + '\n')
        optimiser_options = {
            'gtol' : gradient_tolerance
        }
        result = minimize(obj_func, initial_parameter_guesses, method="BFGS", args=(mixer_list, ham_layers), options=optimiser_options)

        output_writer.write('\tOptimisation completed wih following outcome:\n')
        output_writer.write('\t\tNumber of iterations performed: ' + str(result.nit) + '\n')
        output_writer.write('\t\tNumber of expectation evaluations performed: ' + str(result.nfev) + '\n')
        output_writer.write('\t\tSuccess: ' + str(result.success) + '\n')
        output_writer.write('\t\tOptimiser message: ' + str(result.message) + '\n')

        parameter_list = list(result.x)
        mixer_params = parameter_list[:curr_layer]
        ham_params = parameter_list[curr_layer:]
        mixer_params_string = '\tOptimised mixer unitary parameters: '
        for param in mixer_params:
            mixer_params_string += f'{param:.3}' + ', '
        ham_params_string = '\tOptimised Hamiltonian unitary parameters: '
        for param in ham_params:
            ham_params_string += f'{param:.3}' + ', '

        output_writer.write(mixer_params_string[:-2] + '\n')
        output_writer.write(ham_params_string[:-2] + '\n')
        curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers)
        curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
        ham_approx_ratios.append(curr_ham_estimate)
        cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
        cut_approx_ratios.append(cut_approx_ratio)
        output_writer.write('\nCurrent Cut Approximation Ratio: ' + str(cut_approx_ratio))

        # check convergence
        if curr_layer == max_depth or abs(ham_approx_ratios[-1]-ham_approx_ratios[-2]) < etol:
            not_converged = False

        output_writer.write('\n\n')

    for layer in range(len(ham_approx_ratios)):
        ham_approx_ratios[layer] /= max_ham_value

    data = {
        'cut_approx_ratios' : cut_approx_ratios,
        'ham_approx_ratios' : ham_approx_ratios,
        'best_mixers' : mixer_list,
        'best_mixer_parameters' : mixer_params,
        'best_ham_parameters' : ham_params,
        'all_mixers' : all_mixers_per_layer_dict
    }

    return data

def run_adapt_qaoa_noisy_silent(graph, pauli_ops_dict, gradient_ops_dict, max_depth, beta_0=0.0, gamma_0 = 0.01, rel_gtol = 10**-2, output_file = 'adapt_qaoa.txt', gate_error_prob=0.0, etol=-1):
    """
    Method which runs ADAPT-QAOA with noise.

    Parameters:
        graph - Networkx Graph instance of graph on which to run algorithm
        pauli_ops_dict - dictionary of Pauli sparse matrices for unitary generation
        gradient_ops_dict - dictionary of mixers for which gradients will be evaluated
        max_depth - maximum depth of ansatz
        beta_0 - initial parameter guess for mixer unitaries
        gamma_0 - initial parameter guess for Hamiltonian unitaries
        rel_gtol - factor by which to multiply current maximum gradient when
            setting the gtol for the classical optimiser
        gate_error_prob - probability of depolarizing noise error occuring in each CNOT gate
        output_file - .txt file into which to write simulator output
        etol - tolerance for absolute energy expectation improvement (default = -1, i.e., this criterion is not considered)

    Returns:
        results:
            - achieved max-cut approximation ratios at each layer
            - achieved Hamiltonian max eigenvalue approximation ratios at each layer
            - optimal mixers at each layer
            - optimal mixer unitary parameters
            - optimal Hamiltonian unitary parameters
            - dictionary of gradient magnitudes for each mixer at each layer
    """

    output_writer = open(output_file, 'w')

    if not isinstance(graph, Graph):
        raise Exception("Error - passed graph is not instance of Networkx Graph class!")

    max_cut_solution = useful_methods.find_optimal_cut(graph)
    max_cut_value = max_cut_solution[1]
    max_ham_value = max_cut_solution[2]
    ham_offset = max_cut_value - max_ham_value

    hamiltonian = build_operators.cut_hamiltonian(graph)

    mixer_params = []
    mixer_list = []
    ham_params = []
    ham_layers = []
    ham_approx_ratios = []
    cut_approx_ratios = []
    all_mixers_per_layer_dict = {}

    curr_layer = 0
    curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers, noisy=True, noise_prob=gate_error_prob)
    curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
    cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
    output_writer.write('Initial Cut Approximation Ratio: ' + str(cut_approx_ratio) + '\n\n')
    ham_approx_ratios.append(curr_ham_estimate)
    cut_approx_ratios.append(cut_approx_ratio)

    def obj_func(parameter_values, mixers, ham_unitary_layers):
        
        no_ham_layers = len(ham_unitary_layers)
        no_params = len(parameter_values)
        dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, parameter_values[:(no_params-no_ham_layers)], mixers, parameter_values[(no_params-no_ham_layers):], pauli_ops_dict, ham_unitary_layers, noisy=True, noise_prob=gate_error_prob)
        expectation_value = (hamiltonian * dens_mat).trace().real
        return expectation_value * (-1.0)

    not_converged = True
    while not_converged:

        curr_layer += 1
        output_writer.write("Finding Best Mixer for layer " + str(curr_layer) + "...\n")
        all_mixer_gradients = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=True, noisy=True, noise_prob=gate_error_prob)
        

        mixer_list.append(all_mixer_gradients[0][0])
        gradient_tolerance = all_mixer_gradients[0][1] * rel_gtol
        output_writer.write("\tBest mixer is " + str(all_mixer_gradients[0][0]) + " with gradient magnitude " + str(all_mixer_gradients[0][1]) + '\n\n')

        if not all_mixers_per_layer_dict:

            all_mixers_per_layer_dict = {x[0]:[x[1]] for x in all_mixer_gradients}

        else:

            for mixer in all_mixer_gradients:

                all_mixers_per_layer_dict[mixer[0]].append(mixer[1])

        output_writer.write("Optimising layer " + str(curr_layer) + "...\n")
        initial_parameter_guesses = mixer_params + [beta_0] + ham_params + [gamma_0]
        ham_layers.append(curr_layer)
        initial_parameters_string = '\tInitial parameter guesses: '
        for param in initial_parameter_guesses:
            initial_parameters_string += f'{param:.3}' + ', '
        output_writer.write(initial_parameters_string[:-2] + '\n')
        optimiser_options = {
            'gtol' : gradient_tolerance
        }
        result = minimize(obj_func, initial_parameter_guesses, method="BFGS", args=(mixer_list, ham_layers), options=optimiser_options)

        output_writer.write('\tOptimisation completed wih following outcome:\n')
        output_writer.write('\t\tNumber of iterations performed: ' + str(result.nit) + '\n')
        output_writer.write('\t\tNumber of expectation evaluations performed: ' + str(result.nfev) + '\n')
        output_writer.write('\t\tSuccess: ' + str(result.success) + '\n')
        output_writer.write('\t\tOptimiser message: ' + str(result.message) + '\n')

        parameter_list = list(result.x)
        mixer_params = parameter_list[:curr_layer]
        ham_params = parameter_list[curr_layer:]
        mixer_params_string = '\tOptimised mixer unitary parameters: '
        for param in mixer_params:
            mixer_params_string += f'{param:.3}' + ', '
        ham_params_string = '\tOptimised Hamiltonian unitary parameters: '
        for param in ham_params:
            ham_params_string += f'{param:.3}' + ', '

        output_writer.write(mixer_params_string[:-2] + '\n')
        output_writer.write(ham_params_string[:-2] + '\n')
        curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers, noisy=True, noise_prob=gate_error_prob)
        curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
        ham_approx_ratios.append(curr_ham_estimate)
        cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
        cut_approx_ratios.append(cut_approx_ratio)
        output_writer.write('\nCurrent Cut Approximation Ratio: ' + str(cut_approx_ratio))

        # check convergence
        if curr_layer == max_depth or abs(ham_approx_ratios[-1]-ham_approx_ratios[-2]) < etol:
            not_converged = False

        output_writer.write('\n\n')

    for layer in range(len(ham_approx_ratios)):
        ham_approx_ratios[layer] /= max_ham_value

    data = {
        'cut_approx_ratios' : cut_approx_ratios,
        'ham_approx_ratios' : ham_approx_ratios,
        'best_mixers' : mixer_list,
        'best_mixer_parameters' : mixer_params,
        'best_ham_parameters' : ham_params,
        'all_mixers' : all_mixers_per_layer_dict
    }

    return data

def run_dynamic_adapt_qaoa_silent(graph, pauli_ops_dict, gradient_ops_dict, pauli_mixers_split_ops_dict, max_depth, beta_0=0.0, gamma_tilde = 0.1, rel_gtol = 10**-2, delta_1= 0.0, delta_2=1e-8, output_file = 'dynamic_adapt_qaoa.txt', etol=-1):
    """
    Method which runs Dynamic ADAPT-QAOA.

    Parameters:
        graph - Networkx Graph instance of graph on which to run algorithm
        pauli_ops_dict - dictionary of Pauli sparse matrices for unitary generation
        gradient_ops_dict - dictionary of mixers for which gradients will be evaluated
        pauli_mixers_split_ops_dict - dictionary of Pauli sparse matrices for unitary generation split into
            commuting and anti-commuting parts w.r.t. the Hamiltonian
        max_depth - maximum depth of ansatz
        beta_0 - initial parameter guess for mixer unitaries
        gamma_tilde - initial parameter guess for Hamiltonian unitaries
        rel_gtol - factor by which to multiply current maximum gradient when
            setting the gtol for the classical optimiser
        delta_1, delta_2 - error tolerances for Hamiltonian unitary removal check
        output_file - .txt file into which to write simulator output
        etol - tolerance for absolute energy expectation improvement (default = -1, i.e., this criterion is not considered)

    Returns:
        results:
            - achieved max-cut approximation ratios at each layer
            - achieved Hamiltonian ground state approximation ratios at each layer
            - optimal mixers at each layer
            - optimal mixer unitary parameters
            - optimal Hamiltonian unitary parameters
            - layer numbers at which cost unitaties are included
            - dictionary of gradient magnitudes for each mixer at each layer
    """

    output_writer = open(output_file, 'w')

    if not isinstance(graph, Graph):
        raise Exception("Error - passed graph is not instance of Networkx Graph class!")

    max_cut_solution = useful_methods.find_optimal_cut(graph)
    max_cut_value = max_cut_solution[1]
    max_ham_value = max_cut_solution[2]
    ham_offset = max_cut_value - max_ham_value

    hamiltonian = build_operators.cut_hamiltonian(graph)

    mixer_params = []
    mixer_list = []
    ham_params = []
    ham_layers = []
    ham_approx_ratios = []
    cut_approx_ratios = []
    all_mixers_per_layer_dict = {}

    curr_layer = 0
    curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers)
    curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
    cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
    output_writer.write('Initial Cut Approximation Ratio: ' + str(cut_approx_ratio) + '\n\n')
    ham_approx_ratios.append(curr_ham_estimate / max_ham_value)
    cut_approx_ratios.append(cut_approx_ratio)

    def obj_func(parameter_values, mixers, ham_unitary_layers):
        
        no_ham_layers = len(ham_unitary_layers)
        no_params = len(parameter_values)
        dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, parameter_values[:(no_params-no_ham_layers)], mixers, parameter_values[(no_params-no_ham_layers):], pauli_ops_dict, ham_unitary_layers)
        expectation_value = (hamiltonian * dens_mat).trace().real
        return expectation_value * (-1.0)

    prev_best_mixer = None # cannot append the same mixer consecutive times in this ansatz if there is no Hamiltonian unitary in the current layer

    not_converged = True
    while not_converged:

        curr_layer += 1

        output_writer.write("Finding Best Mixer for layer " + str(curr_layer) + "...\n")
        all_mixer_gradients = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=False)

        best_mixer = all_mixer_gradients[0][0]
        best_mixer_gradient = all_mixer_gradients[0][1]
        if curr_layer > 1 and prev_best_mixer == best_mixer:

            best_mixer = all_mixer_gradients[1][0]
            best_mixer_gradient = all_mixer_gradients[1][1]

        output_writer.write('\tThe best mixer for layer ' + str(curr_layer) + ' with no Hamiltonian unitary is ' + best_mixer + ' with a gradient of ' + str(best_mixer_gradient) + '\n')

        if 'standard' not in best_mixer: # can only perform analysis below for single Pauli string mixers

            # determine whether having not Hamiltonian unitary would correspond to a local maximum
            expectations = useful_methods.calculate_mixer_expectations(curr_dens_mat, best_mixer, pauli_mixers_split_ops_dict, pauli_ops_dict, atol=1e-8)
            output_writer.write('\t<iMH_a> = ' + str(expectations['iMH_a']) + '\n')
            output_writer.write('\t<MH_a^2> = ' + str(expectations['MH_a^2']) + '\n')
            output_writer.write('\t<iMH_a^3> = ' + str(expectations['iMH_a^3']) + '\n')
            hessian = 16 * expectations['iMH_a^3'] * expectations['iMH_a']

            if abs(expectations['MH_a^2']) <= delta_1 and hessian - delta_2 > 0:

                use_ham_unitary = False
                output_writer.write('\tA maximum occurs when using no Hamiltonian unitary so we remove it for this layer!\n')
            
            else:

                use_ham_unitary = True
                output_writer.write('\tIt is unclear whether a maximum occurs when using no Hamiltonian unitary so we will add it to the layer!\n')
                all_mixer_gradients_positive = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=use_ham_unitary, gamma_0=gamma_tilde)
                all_mixer_gradients_negative = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=use_ham_unitary, gamma_0=-1 * gamma_tilde)
                if all_mixer_gradients_negative[0][1] > all_mixer_gradients_positive[0][1]:
                    all_mixer_gradients = all_mixer_gradients_negative
                    gamma_guess = -1.0 * gamma_tilde
                else:
                    all_mixer_gradients = all_mixer_gradients_positive
                    gamma_guess = gamma_tilde
                best_mixer = all_mixer_gradients[0][0]
                best_mixer_gradient = all_mixer_gradients[0][1]
                output_writer.write('\tThe new best mixer for layer ' + str(curr_layer) + ' with a Hamiltonian unitary is ' + best_mixer + ' with a gradient of ' + str(best_mixer_gradient) + '\n')
        
        mixer_list.append(best_mixer)
        prev_best_mixer = best_mixer

        if not all_mixers_per_layer_dict:

            all_mixers_per_layer_dict = {x[0]:[x[1]] for x in all_mixer_gradients}

        else:

            for mixer in all_mixer_gradients:

                all_mixers_per_layer_dict[mixer[0]].append(mixer[1])

        output_writer.write("\nOptimising layer " + str(curr_layer) + "...\n")
        if not use_ham_unitary:
            initial_parameter_guesses = mixer_params + [beta_0] + ham_params
        else:
            initial_parameter_guesses = mixer_params + [beta_0] + ham_params + [gamma_guess]
            ham_layers.append(curr_layer)
        initial_parameters_string = '\tInitial parameter guesses: '
        for param in initial_parameter_guesses:
            initial_parameters_string += f'{param:.3}' + ', '
        output_writer.write(initial_parameters_string[:-2] + '\n')
        optimiser_options = {
            'gtol' : best_mixer_gradient * rel_gtol,
        }
        result = minimize(obj_func, initial_parameter_guesses, method='BFGS', args=(mixer_list, ham_layers), options=optimiser_options)

        output_writer.write('\tOptimisation completed wih following outcome:\n')
        output_writer.write('\t\tNumber of iterations performed: ' + str(result.nit) + '\n')
        output_writer.write('\t\tNumber of expectation evaluations performed: ' + str(result.nfev) + '\n')
        output_writer.write('\t\tSuccess: ' + str(result.success) + '\n')
        output_writer.write('\t\tOptimiser message: ' + str(result.message) + '\n')

        parameter_list = list(result.x)

        mixer_params = parameter_list[:curr_layer]
        ham_params = parameter_list[curr_layer:]
        mixer_params_string = '\tOptimised mixer unitary parameters: '
        for param in mixer_params:
            mixer_params_string += f'{param:.3}' + ', '
        ham_params_string = '\tOptimised Hamiltonian unitary parameters: '
        for param in ham_params:
            ham_params_string += f'{param:.3}' + ', '

        output_writer.write(mixer_params_string[:-2] + '\n')
        output_writer.write(ham_params_string[:-2] + '\n')

        curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers)
        curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
        ham_approx_ratios.append(curr_ham_estimate / max_ham_value)
        cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
        cut_approx_ratios.append(cut_approx_ratio)
        output_writer.write('\nCurrent Cut Approximation Ratio: ' + str(cut_approx_ratio))

        # check convergence
        if curr_layer == max_depth or abs(ham_approx_ratios[-1]-ham_approx_ratios[-2]) < etol:
            not_converged = False

        output_writer.write('\n\n')


    data = {
        'cut_approx_ratios' : cut_approx_ratios,
        'ham_approx_ratios' : ham_approx_ratios,
        'best_mixers' : mixer_list,
        'best_mixer_parameters' : mixer_params,
        'best_ham_parameters' : ham_params,
        'ham_unitary_layers' : ham_layers,
        'all_mixers' : all_mixers_per_layer_dict
    }

    return data

def run_dynamic_adapt_qaoa_noisy_silent(graph, pauli_ops_dict, gradient_ops_dict, pauli_mixers_split_ops_dict, max_depth, beta_0=0.0, gamma_tilde = 0.1, rel_gtol = 10**-2, delta_1 = 0.0, delta_2=1e-8, output_file = 'dynamic_adapt_qaoa.txt', gate_error_prob=0.0, etol=-1):
    """
    Method which runs Dynamic ADAPT-QAOA with noise.

    Parameters:
        graph - Networkx Graph instance of graph on which to run algorithm
        pauli_ops_dict - dictionary of Pauli sparse matrices for unitary generation
        gradient_ops_dict - dictionary of mixers for which gradients will be evaluated
        pauli_mixers_split_ops_dict - dictionary of Pauli sparse matrices for unitary generation split into
            commuting and anti-commuting parts w.r.t. the Hamiltonian
        max_depth - maximum depth of ansatz
        beta_0 - initial parameter guess for mixer unitaries
        gamma_0 - initial parameter guess for Hamiltonian unitaries
        rel_gtol - factor by which to multiply current maximum gradient when
            setting the gtol for the classical optimiser
        delta_1, delta_2 - error tolerances for Hamiltonian unitary removal check
        gate_error_prob - probability of depolarizing noise error occuring in each CNOT gate
        output_file - .txt file into which to write simulator output
        etol - tolerance for absolute energy expectation improvement (default = -1, i.e., this criterion is not considered)

    Returns:
        results:
            - achieved max-cut approximation ratios at each layer
            - achieved Hamiltonian ground state approximation ratios at each layer
            - optimal mixers at each layer
            - optimal mixer unitary parameters
            - optimal Hamiltonian unitary parameters
            - layer numbers at which cost unitaties are included
            - dictionary of gradient magnitudes for each mixer at each layer
    """

    output_writer = open(output_file, 'w')

    if not isinstance(graph, Graph):
        raise Exception("Error - passed graph is not instance of Networkx Graph class!")

    max_cut_solution = useful_methods.find_optimal_cut(graph)
    max_cut_value = max_cut_solution[1]
    max_ham_value = max_cut_solution[2]
    ham_offset = max_cut_value - max_ham_value

    hamiltonian = build_operators.cut_hamiltonian(graph)

    mixer_params = []
    mixer_list = []
    ham_params = []
    ham_layers = []
    ham_approx_ratios = []
    cut_approx_ratios = []
    all_mixers_per_layer_dict = {}

    curr_layer = 0
    curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers, noisy=True, noise_prob=gate_error_prob)
    curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
    cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
    output_writer.write('Initial Cut Approximation Ratio: ' + str(cut_approx_ratio) + '\n\n')
    ham_approx_ratios.append(curr_ham_estimate / max_ham_value)
    cut_approx_ratios.append(cut_approx_ratio)

    def obj_func(parameter_values, mixers, ham_unitary_layers):
        
        no_ham_layers = len(ham_unitary_layers)
        no_params = len(parameter_values)
        dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, parameter_values[:(no_params-no_ham_layers)], mixers, parameter_values[(no_params-no_ham_layers):], pauli_ops_dict, ham_unitary_layers, noisy=True, noise_prob=gate_error_prob)
        expectation_value = (hamiltonian * dens_mat).trace().real
        return expectation_value * (-1.0)

    prev_best_mixer = None # cannot append the same mixer consecutive times in this ansatz if there is no Hamiltonian unitary in the current layer

    not_converged = True
    while not_converged:

        curr_layer += 1

        output_writer.write("Finding Best Mixer for layer " + str(curr_layer) + "...\n")
        all_mixer_gradients = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=False, noisy=True, noise_prob=gate_error_prob)

        best_mixer = all_mixer_gradients[0][0]
        best_mixer_gradient = all_mixer_gradients[0][1]
        if curr_layer > 1 and prev_best_mixer == best_mixer:

            best_mixer = all_mixer_gradients[1][0]
            best_mixer_gradient = all_mixer_gradients[1][1]

        output_writer.write('\tThe best mixer for layer ' + str(curr_layer) + ' with no Hamiltonian unitary is ' + best_mixer + ' with a gradient of ' + str(best_mixer_gradient) + '\n')

        if 'standard' not in best_mixer: # can only perform analysis below for single Pauli string mixers

            # determine whether having not Hamiltonian unitary would correspond to a local maximum
            expectations = useful_methods.calculate_mixer_expectations(curr_dens_mat, best_mixer, pauli_mixers_split_ops_dict, pauli_ops_dict, atol=1e-8)
            output_writer.write('\t<iMH_a> = ' + str(expectations['iMH_a']) + '\n')
            output_writer.write('\t<MH_a^2> = ' + str(expectations['MH_a^2']) + '\n')
            output_writer.write('\t<iMH_a^3> = ' + str(expectations['iMH_a^3']) + '\n')
            hessian = 16 * expectations['iMH_a^3'] * expectations['iMH_a']

            if abs(expectations['MH_a^2']) <= delta_1 and hessian - delta_2 > 0:

                use_ham_unitary = False
                output_writer.write('\tA maximum occurs when using no Hamiltonian unitary so we remove it for this layer!\n')
            
            else:

                use_ham_unitary = True
                output_writer.write('\tIt is unclear whether a maximum occurs when using no Hamiltonian unitary so we will add it to the layer!\n')
                all_mixer_gradients_positive = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=use_ham_unitary, gamma_0=gamma_tilde, noisy=True, noise_prob=gate_error_prob)
                all_mixer_gradients_negative = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=use_ham_unitary, gamma_0=-1 * gamma_tilde, noisy=True, noise_prob=gate_error_prob)
                if all_mixer_gradients_negative[0][1] > all_mixer_gradients_positive[0][1]:
                    all_mixer_gradients = all_mixer_gradients_negative
                    gamma_guess = -1.0 * gamma_tilde
                else:
                    all_mixer_gradients = all_mixer_gradients_positive
                    gamma_guess = gamma_tilde
                best_mixer = all_mixer_gradients[0][0]
                best_mixer_gradient = all_mixer_gradients[0][1]
                output_writer.write('\tThe new best mixer for layer ' + str(curr_layer) + ' with a Hamiltonian unitary is ' + best_mixer + ' with a gradient of ' + str(best_mixer_gradient) + '\n')
        
        mixer_list.append(best_mixer)
        prev_best_mixer = best_mixer

        if not all_mixers_per_layer_dict:

            all_mixers_per_layer_dict = {x[0]:[x[1]] for x in all_mixer_gradients}

        else:

            for mixer in all_mixer_gradients:

                all_mixers_per_layer_dict[mixer[0]].append(mixer[1])

        output_writer.write("\nOptimising layer " + str(curr_layer) + "...\n")
        if not use_ham_unitary:
            initial_parameter_guesses = mixer_params + [beta_0] + ham_params
        else:
            initial_parameter_guesses = mixer_params + [beta_0] + ham_params + [gamma_guess]
            ham_layers.append(curr_layer)
        initial_parameters_string = '\tInitial parameter guesses: '
        for param in initial_parameter_guesses:
            initial_parameters_string += f'{param:.3}' + ', '
        output_writer.write(initial_parameters_string[:-2] + '\n')
        optimiser_options = {
            'gtol' : best_mixer_gradient * rel_gtol,
        }
        result = minimize(obj_func, initial_parameter_guesses, method='BFGS', args=(mixer_list, ham_layers), options=optimiser_options)

        output_writer.write('\tOptimisation completed wih following outcome:\n')
        output_writer.write('\t\tNumber of iterations performed: ' + str(result.nit) + '\n')
        output_writer.write('\t\tNumber of expectation evaluations performed: ' + str(result.nfev) + '\n')
        output_writer.write('\t\tSuccess: ' + str(result.success) + '\n')
        output_writer.write('\t\tOptimiser message: ' + str(result.message) + '\n')

        parameter_list = list(result.x)

        mixer_params = parameter_list[:curr_layer]
        ham_params = parameter_list[curr_layer:]
        mixer_params_string = '\tOptimised mixer unitary parameters: '
        for param in mixer_params:
            mixer_params_string += f'{param:.3}' + ', '
        ham_params_string = '\tOptimised Hamiltonian unitary parameters: '
        for param in ham_params:
            ham_params_string += f'{param:.3}' + ', '

        output_writer.write(mixer_params_string[:-2] + '\n')
        output_writer.write(ham_params_string[:-2] + '\n')

        curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers, noisy=True, noise_prob=gate_error_prob)
        curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
        ham_approx_ratios.append(curr_ham_estimate / max_ham_value)
        cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
        cut_approx_ratios.append(cut_approx_ratio)
        output_writer.write('\nCurrent Cut Approximation Ratio: ' + str(cut_approx_ratio))

        # check convergence
        if curr_layer == max_depth or abs(ham_approx_ratios[-1]-ham_approx_ratios[-2]) < etol:
            not_converged = False

        output_writer.write('\n\n')


    data = {
        'cut_approx_ratios' : cut_approx_ratios,
        'ham_approx_ratios' : ham_approx_ratios,
        'best_mixers' : mixer_list,
        'best_mixer_parameters' : mixer_params,
        'best_ham_parameters' : ham_params,
        'ham_unitary_layers' : ham_layers,
        'all_mixers' : all_mixers_per_layer_dict
    }

    return data

def run_dynamic_adapt_qaoa_single_mixer_silent(graph, pauli_ops_dict, gradient_ops_dict, pauli_mixers_split_ops_dict, max_depth, beta_0=0.0, gamma_0 = 0.0, rel_gtol = 10**-2, delta_1 = 0.0, delta_2=1e-8, output_file = 'dynamic_adapt_qaoa.txt', etol=-1):
    """
    Method which runs Dynamic ADAPT-QAOA which keeps the originally found mixer
    in the case where the cost unitaries is kept in the current layer.

    Parameters:
        graph - Networkx Graph instance of graph on which to run algorithm
        pauli_ops_dict - dictionary of Pauli sparse matrices for unitary generation
        gradient_ops_dict - dictionary of mixers for which gradients will be evaluated
        pauli_mixers_split_ops_dict - dictionary of Pauli sparse matrices for unitary generation split into
            commuting and anti-commuting parts w.r.t. the Hamiltonian
        max_depth - maximum depth of ansatz
        beta_0 - initial parameter guess for mixer unitaries
        gamma_0 - initial parameter guess for Hamiltonian unitaries
        rel_gtol - factor by which to multiply current maximum gradient when
            setting the gtol for the classical optimiser
        delta_1, delta_2 - error tolerances for Hamiltonian unitary removal check
        gate_error_prob - probability of depolarizing noise error occuring in each CNOT gate
        output_file - .txt file into which to write simulator output
        etol - tolerance for absolute energy expectation improvement (default = -1, i.e., this criterion is not considered)

    Returns:
        results:
            - achieved max-cut approximation ratios at each layer
            - achieved Hamiltonian ground state approximation ratios at each layer
            - optimal mixers at each layer
            - optimal mixer unitary parameters
            - optimal Hamiltonian unitary parameters
            - layer numbers at which cost unitaties are included
            - dictionary of gradient magnitudes for each mixer at each layer
    """

    output_writer = open(output_file, 'w')

    if not isinstance(graph, Graph):
        raise Exception("Error - passed graph is not instance of Networkx Graph class!")

    max_cut_solution = useful_methods.find_optimal_cut(graph)
    max_cut_value = max_cut_solution[1]
    max_ham_value = max_cut_solution[2]
    ham_offset = max_cut_value - max_ham_value

    hamiltonian = build_operators.cut_hamiltonian(graph)

    mixer_params = []
    mixer_list = []
    ham_params = []
    ham_layers = []
    ham_approx_ratios = []
    cut_approx_ratios = []
    all_mixers_per_layer_dict = {}

    curr_layer = 0
    curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers)
    curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
    cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
    output_writer.write('Initial Cut Approximation Ratio: ' + str(cut_approx_ratio) + '\n\n')
    ham_approx_ratios.append(curr_ham_estimate / max_ham_value)
    cut_approx_ratios.append(cut_approx_ratio)

    def obj_func(parameter_values, mixers, ham_unitary_layers):
        
        no_ham_layers = len(ham_unitary_layers)
        no_params = len(parameter_values)
        dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, parameter_values[:(no_params-no_ham_layers)], mixers, parameter_values[(no_params-no_ham_layers):], pauli_ops_dict, ham_unitary_layers)
        expectation_value = (hamiltonian * dens_mat).trace().real
        return expectation_value * (-1.0)

    prev_best_mixer = None # cannot append the same mixer consecutive times in this ansatz if there is no Hamiltonian unitary in the current layer

    not_converged = True
    while not_converged:

        curr_layer += 1

        output_writer.write("Finding Best Mixer for layer " + str(curr_layer) + "...\n")
        all_mixer_gradients = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=False)

        best_mixer = all_mixer_gradients[0][0]
        best_mixer_gradient = all_mixer_gradients[0][1]
        if curr_layer > 1 and prev_best_mixer == best_mixer:

            best_mixer = all_mixer_gradients[1][0]
            best_mixer_gradient = all_mixer_gradients[1][1]

        output_writer.write('\tThe best mixer for layer ' + str(curr_layer) + ' with no Hamiltonian unitary is ' + best_mixer + ' with a gradient of ' + str(best_mixer_gradient) + '\n')

        if 'standard' not in best_mixer: # can only perform analysis below for single Pauli string mixers

            # determine whether having not Hamiltonian unitary would correspond to a local maximum
            expectations = useful_methods.calculate_mixer_expectations(curr_dens_mat, best_mixer, pauli_mixers_split_ops_dict, pauli_ops_dict, atol=1e-8)
            output_writer.write('\t<iMH_a> = ' + str(expectations['iMH_a']) + '\n')
            output_writer.write('\t<MH_a^2> = ' + str(expectations['MH_a^2']) + '\n')
            output_writer.write('\t<iMH_a^3> = ' + str(expectations['iMH_a^3']) + '\n')
            hessian = 16 * expectations['iMH_a^3'] * expectations['iMH_a']

            if abs(expectations['MH_a^2']) <= delta_1 and hessian - delta_2 > 0:

                use_ham_unitary = False
                output_writer.write('\tA maximum occurs when using no Hamiltonian unitary so we remove it for this layer!\n')
            
            else:

                use_ham_unitary = True
                output_writer.write('\tIt is unclear whether a maximum occurs when using no Hamiltonian unitary so we will add it to the layer!\n')
                # all_mixer_gradients_positive = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=use_ham_unitary, gamma_0=gamma_0)
                # all_mixer_gradients_negative = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=use_ham_unitary, gamma_0=-1 * gamma_0)
                # if all_mixer_gradients_negative[0][1] > all_mixer_gradients_positive[0][1]:
                #     all_mixer_gradients = all_mixer_gradients_negative
                #     gamma_guess = -1.0 * gamma_0
                # else:
                #     all_mixer_gradients = all_mixer_gradients_positive
                #     gamma_guess = gamma_0
                # best_mixer = all_mixer_gradients[0][0]
                # best_mixer_gradient = all_mixer_gradients[0][1]
                # output_writer.write('\tThe new best mixer for layer ' + str(curr_layer) + ' with a Hamiltonian unitary is ' + best_mixer + ' with a gradient of ' + str(best_mixer_gradient) + '\n')
        
        mixer_list.append(best_mixer)
        prev_best_mixer = best_mixer

        if not all_mixers_per_layer_dict:

            all_mixers_per_layer_dict = {x[0]:[x[1]] for x in all_mixer_gradients}

        else:

            for mixer in all_mixer_gradients:

                all_mixers_per_layer_dict[mixer[0]].append(mixer[1])

        output_writer.write("\nOptimising layer " + str(curr_layer) + "...\n")
        if not use_ham_unitary:
            initial_parameter_guesses = mixer_params + [beta_0] + ham_params
        else:
            initial_parameter_guesses = mixer_params + [beta_0] + ham_params + [gamma_0]
            ham_layers.append(curr_layer)
        initial_parameters_string = '\tInitial parameter guesses: '
        for param in initial_parameter_guesses:
            initial_parameters_string += f'{param:.3}' + ', '
        output_writer.write(initial_parameters_string[:-2] + '\n')
        optimiser_options = {
            'gtol' : best_mixer_gradient * rel_gtol,
        }
        result = minimize(obj_func, initial_parameter_guesses, method='BFGS', args=(mixer_list, ham_layers), options=optimiser_options)

        output_writer.write('\tOptimisation completed wih following outcome:\n')
        output_writer.write('\t\tNumber of iterations performed: ' + str(result.nit) + '\n')
        output_writer.write('\t\tNumber of expectation evaluations performed: ' + str(result.nfev) + '\n')
        output_writer.write('\t\tSuccess: ' + str(result.success) + '\n')
        output_writer.write('\t\tOptimiser message: ' + str(result.message) + '\n')

        parameter_list = list(result.x)

        mixer_params = parameter_list[:curr_layer]
        ham_params = parameter_list[curr_layer:]
        mixer_params_string = '\tOptimised mixer unitary parameters: '
        for param in mixer_params:
            mixer_params_string += f'{param:.3}' + ', '
        ham_params_string = '\tOptimised Hamiltonian unitary parameters: '
        for param in ham_params:
            ham_params_string += f'{param:.3}' + ', '

        output_writer.write(mixer_params_string[:-2] + '\n')
        output_writer.write(ham_params_string[:-2] + '\n')

        curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers)
        curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
        ham_approx_ratios.append(curr_ham_estimate / max_ham_value)
        cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
        cut_approx_ratios.append(cut_approx_ratio)
        output_writer.write('\nCurrent Cut Approximation Ratio: ' + str(cut_approx_ratio))

        # check convergence
        if curr_layer == max_depth or abs(ham_approx_ratios[-1]-ham_approx_ratios[-2]) < etol:
            not_converged = False

        output_writer.write('\n\n')


    data = {
        'cut_approx_ratios' : cut_approx_ratios,
        'ham_approx_ratios' : ham_approx_ratios,
        'best_mixers' : mixer_list,
        'best_mixer_parameters' : mixer_params,
        'best_ham_parameters' : ham_params,
        'ham_unitary_layers' : ham_layers,
        'all_mixers' : all_mixers_per_layer_dict
    }

    return data

def run_adapt_qaoa_no_cost_unitaries_silent(graph, pauli_ops_dict, gradient_ops_dict, max_depth, beta_0=0.0, rel_gtol = 10**-2, output_file = 'dynamic_adapt_qaoa_no_cost_unitaries.txt', etol=-1):
    """
    Method which runs ADAPT-QAOA with no cost unitaries whatsoever.

    Parameters:
        graph - Networkx Graph instance of graph on which to run algorithm
        pauli_ops_dict - dictionary of Pauli sparse matrices for unitary generation
        gradient_ops_dict - dictionary of mixers for which gradients will be evaluated
        pauli_mixers_split_ops_dict - dictionary of Pauli sparse matrices for unitary generation split into
            commuting and anti-commuting parts w.r.t. the Hamiltonian
        max_depth - maximum depth of ansatz
        beta_0 - initial parameter guess for mixer unitaries
        gamma_0 - initial parameter guess for Hamiltonian unitaries
        rel_gtol - factor by which to multiply current maximum gradient when
            setting the gtol for the classical optimiser
        output_file - .txt file into which to write simulator output
        etol - tolerance for absolute energy expectation improvement (default = -1, i.e., this criterion is not considered)

    Returns:
        results:
            - achieved max-cut approximation ratios at each layer
            - achieved Hamiltonian max eigenvalue approximation ratios at each layer
            - optimal mixers at each layer
            - optimal mixer unitary parameters
            - dictionary of gradient magnitudes for each mixer at each layer
    """

    output_writer = open(output_file, 'w')

    if not isinstance(graph, Graph):
        raise Exception("Error - passed graph is not instance of Networkx Graph class!")

    max_cut_solution = useful_methods.find_optimal_cut(graph)
    max_cut_value = max_cut_solution[1]
    max_ham_value = max_cut_solution[2]
    ham_offset = max_cut_value - max_ham_value

    hamiltonian = build_operators.cut_hamiltonian(graph)

    mixer_params = []
    mixer_list = []
    ham_params = []
    ham_layers = []
    ham_approx_ratios = []
    cut_approx_ratios = []
    all_mixers_per_layer_dict = {}

    curr_layer = 0
    curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers)
    curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
    cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
    output_writer.write('Initial Cut Approximation Ratio: ' + str(cut_approx_ratio) + '\n\n')
    ham_approx_ratios.append(curr_ham_estimate / max_ham_value)
    cut_approx_ratios.append(cut_approx_ratio)

    def obj_func(parameter_values, mixers, ham_unitary_layers):
        
        no_ham_layers = len(ham_unitary_layers)
        no_params = len(parameter_values)
        dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, parameter_values[:(no_params-no_ham_layers)], mixers, parameter_values[(no_params-no_ham_layers):], pauli_ops_dict, ham_unitary_layers)
        expectation_value = (hamiltonian * dens_mat).trace().real
        return expectation_value * (-1.0)

    prev_best_mixer = None # cannot append the same mixer consecutive times in this ansatz if there is no Hamiltonian unitary in the current layer

    not_converged = True
    while not_converged:

        curr_layer += 1

        output_writer.write("Finding Best Mixer for layer " + str(curr_layer) + "...\n")
        all_mixer_gradients = useful_methods.find_mixer_gradients(curr_dens_mat, gradient_ops_dict, pauli_ops_dict, graph, apply_ham_unitary=False)

        best_mixer = all_mixer_gradients[0][0]
        best_mixer_gradient = all_mixer_gradients[0][1]
        if curr_layer > 1 and prev_best_mixer == best_mixer:

            best_mixer = all_mixer_gradients[1][0]
            best_mixer_gradient = all_mixer_gradients[1][1]

        output_writer.write('\tThe best mixer for layer ' + str(curr_layer) + ' with no Hamiltonian unitary is ' + best_mixer + ' with a gradient of ' + str(best_mixer_gradient) + '\n')

        mixer_list.append(best_mixer)
        prev_best_mixer = best_mixer

        if not all_mixers_per_layer_dict:

            all_mixers_per_layer_dict = {x[0]:[x[1]] for x in all_mixer_gradients}

        else:

            for mixer in all_mixer_gradients:

                all_mixers_per_layer_dict[mixer[0]].append(mixer[1])

        output_writer.write("\nOptimising layer " + str(curr_layer) + "...\n")
        initial_parameter_guesses = mixer_params + [beta_0] + ham_params
        initial_parameters_string = '\tInitial parameter guesses: '
        for param in initial_parameter_guesses:
            initial_parameters_string += f'{param:.3}' + ', '
        output_writer.write(initial_parameters_string[:-2] + '\n')
        optimiser_options = {
            'gtol' : best_mixer_gradient * rel_gtol,
        }
        result = minimize(obj_func, initial_parameter_guesses, method='BFGS', args=(mixer_list, ham_layers), options=optimiser_options)

        output_writer.write('\tOptimisation completed wih following outcome:\n')
        output_writer.write('\t\tNumber of iterations performed: ' + str(result.nit) + '\n')
        output_writer.write('\t\tNumber of expectation evaluations performed: ' + str(result.nfev) + '\n')
        output_writer.write('\t\tSuccess: ' + str(result.success) + '\n')
        output_writer.write('\t\tOptimiser message: ' + str(result.message) + '\n')

        parameter_list = list(result.x)

        mixer_params = parameter_list[:curr_layer]
        ham_params = parameter_list[curr_layer:]
        mixer_params_string = '\tOptimised mixer unitary parameters: '
        for param in mixer_params:
            mixer_params_string += f'{param:.3}' + ', '
        ham_params_string = '\tOptimised Hamiltonian unitary parameters: '
        for param in ham_params:
            ham_params_string += f'{param:.3}' + ', '

        output_writer.write(mixer_params_string[:-2] + '\n')
        output_writer.write(ham_params_string[:-2] + '\n')

        curr_dens_mat = build_operators.build_adapt_qaoa_ansatz(graph, mixer_params, mixer_list, ham_params, pauli_ops_dict, ham_layers)
        curr_ham_estimate = (hamiltonian * curr_dens_mat).trace().real
        ham_approx_ratios.append(curr_ham_estimate / max_ham_value)
        cut_approx_ratio = (curr_ham_estimate + ham_offset) / max_cut_value
        cut_approx_ratios.append(cut_approx_ratio)
        output_writer.write('\nCurrent Cut Approximation Ratio: ' + str(cut_approx_ratio))

        # check convergence
        if curr_layer == max_depth or abs(ham_approx_ratios[-1]-ham_approx_ratios[-2]) < etol:
            not_converged = False

        output_writer.write('\n\n')


    data = {
        'cut_approx_ratios' : cut_approx_ratios,
        'ham_approx_ratios' : ham_approx_ratios,
        'best_mixers' : mixer_list,
        'best_mixer_parameters' : mixer_params,
        'all_mixers' : all_mixers_per_layer_dict
    }

    return data

# for standard QAOA one can use the implementation above as it has no output by default

#######################
# classical algorithm #
#######################

def run_goemans_williamson(graph, no_relaxations = 10_000):
    """
    Method which runs the classical SDP algorithm for max cut.

    Parameters:
        graph - networkx Graph instance
        no_relaxations - number of relaxations to perform on the SDP
        output in order to find a cut
    
    Returns:
        approximation_ratio - Maximum cut approximation ratio found as average for all relaxations
        error - error in approximation ratio above
    """

    if not isinstance(graph, Graph):
        raise Exception("Error - passed graph is not an instance of the networkx Graph class!")

    sdp_solution = useful_methods.goemans_williamson(graph)
    max_cut = useful_methods.find_optimal_cut(graph)[1]
    dimension = graph.number_of_nodes()
    approximation_ratios = [None] * no_relaxations
    cut_distribution = {}
    
    for relaxation in range(no_relaxations):

        # generate random direction in space
        random_vector = np.random.randn(dimension)
    
        cut = np.sign(sdp_solution @ random_vector)
        cut_bitstring = ''
        for i in cut:
            if i > 0:
                cut_bitstring += '0'
            else:
                cut_bitstring += '1'
        if cut_bitstring not in cut_distribution:
            cut_distribution[cut_bitstring] = 1
        else:
            cut_distribution[cut_bitstring] += 1
        cut_value = 0

        for pair in list(graph.edges()):
            weight = graph.get_edge_data(*pair)['weight']
            if cut[pair[0]] != cut[pair[1]]:
                cut_value += weight

        approximation_ratios[relaxation] = cut_value / max_cut

    approximation_ratio = np.average(approximation_ratios)
    error = np.std(approximation_ratios) / np.sqrt(no_relaxations)

    return approximation_ratio, error