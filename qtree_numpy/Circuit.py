import cirq
import numpy as np
from functools import reduce

class Circuit:
    def __init__(self):
        """
        create new Circuit
        """
        self.operators = []
        self.circuit = [[]]
        self.layer_idx = 0
        self.qubits = [] # list of integer indexies of qubits
        self.qubit_count = len(self.qubits)
        self.layer_qubits = []
    def append(self,op):
        self.operators.append(op)
        # if some of the qubits indexes from operation are in 
        # current layer, make a new layer
        _l = [x in self.layer_qubits for x in op._qubits]
        if reduce(lambda x,y:x and y,_l):
            self.next_layer()
        self.layer_qubits += op._qubits
        self.circuit[self.layer_idx].append(op)

    def convert_to_cirq(self):
        side_length = int(np.sqrt(self.qubit_count))
        cirq_circuit = cirq.Circuit()
        for layer in self.circuit:
            cirq_circuit.append(op.to_cirq_2d_circ_op(side_length) for op in layer)
        return cirq_circuit

    def next_layer(self):
        """ append an empty list to circuit,
        increment the layer index, clear list of layer qubits
        update qubit count if necesary
        """
        self.circuit.append([])
        self.layer_idx = self.layer_idx + 1
        # qubit count is maximum of qubit size of layer
        # by design, should be number of unique qubit indexes
        if self.qubit_count < len(self.layer_qubits):
            self.qubit_count = len(self.layer_qubits)
        self.layer_qubits = []
