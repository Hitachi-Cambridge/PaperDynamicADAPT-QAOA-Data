import networkx as nx
import numpy as np
from qiskit import quantum_info as qi
from scipy import sparse

class mixer_operator:

    def __init__(self, graph) -> None:

        if not isinstance(graph, nx.classes.graph.Graph):
            raise Exception("Error - passed graph must be instance of networkx Graph class!")

        self.no_qubits = graph.number_of_nodes()
        self.gradient_operator = None
        self.type = ""

    def find_exact_gradient(self, dens_mat):

        if not isinstance(dens_mat, sparse._csr.csr_matrix):
            raise Exception("Error - passed density matrix must be instance of the scipy csrmatrix class!")

        expectation_value = (self.gradient_operator * dens_mat).trace()
        return expectation_value.real

    def __str__(self) -> str:
        return self.type

class standard_mixer_x_gates(mixer_operator):

    def __init__(self, graph) -> None:

        super().__init__(graph)
        self.type = "Standard Mixer with X gates"
        

    def find_exact_gradient(self, dens_mat):

        raise Exception("Error - exact gradient for standard-type mixers must be found from gradients for single-qubit mixers!")


class standard_mixer_y_gates(mixer_operator):

    def __init__(self, graph) -> None:

        super().__init__(graph)
        self.type = "Standard Mixer with Y gates"

    def find_exact_gradient(self, dens_mat):

        raise Exception("Error - exact gradient for standard-type mixers must be found from gradients for single-qubit mixers!")

class X_mixer(mixer_operator):

    def __init__(self, graph, x_qubit) -> None:

        super().__init__(graph)
        self.x_qubit = x_qubit
        no_ops = graph.degree[self.x_qubit]
        pauli_strings = [None] * no_ops
        coeffs = [None] * no_ops
        index = 0
        for i in range(self.no_qubits):

            if i == self.x_qubit:
                continue
            if graph.get_edge_data(i, self.x_qubit) == None:
                continue

            if i < self.x_qubit:
                tmp_str = 'I' * (i) + 'Z' + 'I' * (self.x_qubit-i-1) + 'Y' + 'I' * (self.no_qubits - self.x_qubit - 1)
            elif i > self.x_qubit:
                tmp_str = 'I' * (self.x_qubit) + 'Y' + 'I' * (i - self.x_qubit - 1) + 'Z' + 'I' * (self.no_qubits - i - 1)
            tmp_str = tmp_str[::-1]
            pauli_strings[index] = tmp_str
            coeffs[index] = (-1.0) * graph.get_edge_data(i, self.x_qubit)['weight']
            index += 1

        self.gradient_operator = sparse.csr_matrix(qi.SparsePauliOp(pauli_strings, np.array(coeffs)).to_matrix())
        self.type = "X" + str(self.x_qubit) + " Mixer"

class Y_mixer(mixer_operator):

    def __init__(self, graph, y_qubit) -> None:
        
        super().__init__(graph)
        self.y_qubit = y_qubit
        no_ops = graph.degree[self.y_qubit]
        pauli_strings = [None] * no_ops
        coeffs = [None] * no_ops
        index = 0
        for i in range(self.no_qubits):

            if i == self.y_qubit:
                continue
            if graph.get_edge_data(i, self.y_qubit) == None:
                continue
            
            if i < self.y_qubit:
                tmp_str = 'I' * (i) + 'Z' + 'I' * (self.y_qubit-i-1) + 'X' + 'I' * (self.no_qubits - self.y_qubit - 1)
            elif i > self.y_qubit:
                tmp_str = 'I' * (self.y_qubit) + 'X' + 'I' * (i - self.y_qubit - 1) + 'Z' + 'I' * (self.no_qubits - i - 1)
            tmp_str = tmp_str[::-1]
            pauli_strings[index] = tmp_str
            coeffs[index] = graph.get_edge_data(i, self.y_qubit)['weight']
            index += 1

        self.gradient_operator = sparse.csr_matrix(qi.SparsePauliOp(pauli_strings, np.array(coeffs)).to_matrix())
        self.type = "Y" + str(self.y_qubit) + " Mixer"

class XZ_mixer(mixer_operator):

    def __init__(self, graph, x_qubit, z_qubit) -> None:

        if x_qubit == z_qubit:
            raise Exception("Error - Pauli's must act on different qubits!")

        super().__init__(graph)
        self.x_qubit = x_qubit
        self.z_qubit = z_qubit
        no_ops = graph.degree[self.x_qubit]
        pauli_strings = [None] * no_ops
        coeffs = [None] * no_ops
        index = 0
        for i in range(self.no_qubits):

            if i == self.x_qubit:
                continue
            if graph.get_edge_data(i, self.x_qubit) == None:
                continue
            
            paulis = ['I'] * self.no_qubits
            paulis[self.x_qubit] = 'Y'
            if i != self.z_qubit:
                paulis[i] = 'Z'
                paulis[self.z_qubit] = 'Z'
            tmp_str = ''.join(paulis)
            tmp_str = tmp_str[::-1]
            pauli_strings[index] = tmp_str
            coeffs[index] = graph.get_edge_data(i, self.x_qubit)['weight']
            index += 1

        self.gradient_operator = sparse.csr_matrix(qi.SparsePauliOp(pauli_strings, np.array(coeffs)).to_matrix())
        self.type = "X" + str(self.x_qubit) + "Z" + str(self.z_qubit) + " Mixer"

class YZ_mixer(mixer_operator):

    def __init__(self, graph, y_qubit, z_qubit) -> None:

        if y_qubit == z_qubit:
            raise Exception("Error - Pauli's must act on different qubits!")

        super().__init__(graph)
        self.y_qubit = y_qubit
        self.z_qubit = z_qubit
        no_ops = graph.degree[self.y_qubit]
        pauli_strings = [None] * no_ops
        coeffs = [None] * no_ops
        index = 0
        for i in range(self.no_qubits):

            if i == self.y_qubit:
                continue
            if graph.get_edge_data(i, self.y_qubit) == None:
                continue
            
            paulis = ['I'] * self.no_qubits
            paulis[self.y_qubit] = 'X'
            if i != self.z_qubit:
                paulis[i] = 'Z'
                paulis[self.z_qubit] = 'Z'
            tmp_str = ''.join(paulis)
            tmp_str = tmp_str[::-1]
            pauli_strings[index] = tmp_str
            coeffs[index] = (-1.0) * graph.get_edge_data(i, self.y_qubit)['weight']
            index += 1

        self.gradient_operator = sparse.csr_matrix(qi.SparsePauliOp(pauli_strings, np.array(coeffs)).to_matrix())
        self.type = "Y" + str(self.y_qubit) + "Z" + str(self.z_qubit) + " Mixer"

class XY_mixer(mixer_operator):

    def __init__(self, graph, x_qubit, y_qubit) -> None:

        super().__init__(graph)
        self.x_qubit = x_qubit
        self.y_qubit = y_qubit
        no_ops_x = graph.degree[self.x_qubit]
        no_ops_y = graph.degree[self.y_qubit]
        common_edge = int(graph.get_edge_data(self.x_qubit, self.y_qubit) != None)
        pauli_strings = [None] * (no_ops_x + no_ops_y - 2 * common_edge)
        coeffs = [None] * (no_ops_x + no_ops_y - 2 * common_edge)
        index = 0
        for i in range(self.no_qubits):

            if i == self.x_qubit or i == self.y_qubit:
                continue

            if graph.get_edge_data(i, self.x_qubit) != None:

                paulis = ['I'] * self.no_qubits
                paulis[self.x_qubit] = 'Y'
                paulis[i] = 'Z'
                paulis[self.y_qubit] = 'Y'
                tmp_str = ''.join(paulis)
                tmp_str = tmp_str[::-1]
                pauli_strings[index] = tmp_str
                coeffs[index] = graph.get_edge_data(i, self.x_qubit)['weight']
                index += 1
            
            if graph.get_edge_data(i, self.y_qubit) != None:

                paulis = ['I'] * self.no_qubits
                paulis[self.x_qubit] = 'X'
                paulis[i] = 'Z'
                paulis[self.y_qubit] = 'X'
                tmp_str = ''.join(paulis)
                tmp_str = tmp_str[::-1]
                pauli_strings[index] = tmp_str
                coeffs[index] = (-1.0) * graph.get_edge_data(i, self.y_qubit)['weight']
                index += 1

        self.gradient_operator = sparse.csr_matrix(qi.SparsePauliOp(pauli_strings, np.array(coeffs)).to_matrix())
        self.type = "X" + str(self.x_qubit) + "Y" + str(self.y_qubit) + " Mixer"

class YY_mixer(mixer_operator):

    def __init__(self, graph, qubit_k, qubit_l) -> None:

        super().__init__(graph)
        self.qubit_k = qubit_k
        self.qubit_l = qubit_l
        no_ops_k = graph.degree[self.qubit_k]
        no_ops_l = graph.degree[self.qubit_l]
        common_edge = int(graph.get_edge_data(self.qubit_k, self.qubit_l) != None)
        pauli_strings = [None] * (no_ops_k + no_ops_l - 2 * common_edge)
        coeffs = [None] * (no_ops_k + no_ops_l - 2 * common_edge)
        index = 0
        for i in range(self.no_qubits):

            if i == self.qubit_k or i == self.qubit_l:
                continue

            if graph.get_edge_data(i, self.qubit_k) != None:

                paulis = ['I'] * self.no_qubits
                paulis[self.qubit_k] = 'X'
                paulis[i] = 'Z'
                paulis[self.qubit_l] = 'Y'
                tmp_str = ''.join(paulis)
                tmp_str = tmp_str[::-1]
                pauli_strings[index] = tmp_str
                coeffs[index] = (-1.0) * graph.get_edge_data(i, self.qubit_k)['weight']
                index += 1
            
            if graph.get_edge_data(i, self.qubit_l) != None:

                paulis = ['I'] * self.no_qubits
                paulis[self.qubit_k] = 'Y'
                paulis[i] = 'Z'
                paulis[self.qubit_l] = 'X'
                tmp_str = ''.join(paulis)
                tmp_str = tmp_str[::-1]
                pauli_strings[index] = tmp_str
                coeffs[index] = (-1.0) * graph.get_edge_data(i, self.qubit_l)['weight']
                index += 1

        self.gradient_operator = sparse.csr_matrix(qi.SparsePauliOp(pauli_strings, np.array(coeffs)).to_matrix())
        self.type = "Y" + str(self.qubit_k) + "Y" + str(self.qubit_l) + " Mixer"

class XX_mixer(mixer_operator):

    def __init__(self, graph, qubit_k, qubit_l) -> None:

        super().__init__(graph)
        self.qubit_k = qubit_k
        self.qubit_l = qubit_l
        no_ops_k = graph.degree[self.qubit_k]
        no_ops_l = graph.degree[self.qubit_l]
        common_edge = int(graph.get_edge_data(self.qubit_k, self.qubit_l) != None)
        pauli_strings = [None] * (no_ops_k + no_ops_l - 2 * common_edge)
        coeffs = [None] * (no_ops_k + no_ops_l - 2 * common_edge)
        index = 0
        for i in range(self.no_qubits):

            if i == self.qubit_k or i == self.qubit_l:
                continue

            if graph.get_edge_data(i, self.qubit_k) != None:

                paulis = ['I'] * self.no_qubits
                paulis[self.qubit_k] = 'Y'
                paulis[i] = 'Z'
                paulis[self.qubit_l] = 'X'
                tmp_str = ''.join(paulis)
                tmp_str = tmp_str[::-1]
                pauli_strings[index] = tmp_str
                coeffs[index] = graph.get_edge_data(i, self.qubit_k)['weight']
                index += 1
            
            if graph.get_edge_data(i, self.qubit_l) != None:

                paulis = ['I'] * self.no_qubits
                paulis[self.qubit_k] = 'X'
                paulis[i] = 'Z'
                paulis[self.qubit_l] = 'Y'
                tmp_str = ''.join(paulis)
                tmp_str = tmp_str[::-1]
                pauli_strings[index] = tmp_str
                coeffs[index] = graph.get_edge_data(i, self.qubit_l)['weight']
                index += 1

        self.gradient_operator = sparse.csr_matrix(qi.SparsePauliOp(pauli_strings, np.array(coeffs)).to_matrix())
        self.type = "X" + str(self.qubit_k) + "X" + str(self.qubit_l) + " Mixer"