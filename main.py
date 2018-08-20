import argparse,re
import numpy as np
from src.logging import get_logger
log = get_logger()
from qtree_numpy import Simulator, Circuit
from qtree_numpy.operators import *


OP = qOperation()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('circuitfile', help='file with circuit')
    parser.add_argument('target_state',
                        help='state x against which amplitude is computed')
    args = parser.parse_args()
    start_simulation(args.circuitfile,args.target_state)

def start_simulation(circuit_file,target_state):
    circuit = read_circuit_file(circuit_file)
    cirq_circuit = circuit.convert_to_cirq()
    sim = Simulator()
    final_state_qtree = sim.simulate(
        circuit,
        parallel=True,
        #graph_model_plot='gr.png'
    )

    cirq_sim = cirq.google.XmonSimulator()
    cirq_result =cirq_sim.simulate(cirq_circuit)
    print(cirq_circuit)
    print("cirq ",cirq_result.final_state.round(4))
    print("qtree", final_state_qtree.round(4))

def read_circuit_file(filename, max_depth=None):
    log.info("reading file {}".format(filename))
    circuit = Circuit()
    with open(filename, "r") as fp:
        qubit_count = int(fp.readline())
        current_layer = 0
        log.info("There should be {:d} qubits in circuit".format(qubit_count))
        for line in fp:
            m = re.search(r'(?P<layer>[0-9]+) (?=[a-z])', line)
            # Read circuit layer by layer
            layer_num = int(m.group('layer'))
            if m is None:
                raise Exception(
                    "file format error at line {}".format(line))

            if layer_num > current_layer:
                circuit.next_layer()
                current_layer = layer_num
            op_str = line[m.end():]
            op = OP.factory(op_str)
            circuit.append(op)
    if circuit.qubit_count!=qubit_count:
        log.warn("Cirquit has {:d} qubits without any operator on them!"
                 .format(qubit_count-circuit.qubit_count)
                )
    circuit.qubit_count = qubit_count
    print(circuit.circuit)
    return circuit

if __name__=="__main__":
    main()
