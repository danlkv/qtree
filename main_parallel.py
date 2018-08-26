import argparse,re
import numpy as np
from src.logging import get_logger
log = get_logger()
from qtree_numpy import Simulator, Circuit
from qtree_numpy.operators import *
from mpi4py import MPI
import time


OP = qOperation()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('circuitfile', help='file with circuit')
    parser.add_argument('target_state',
                        help='state x against which amplitude is computed')
    args = parser.parse_args()
    start_simulation(args.circuitfile,args.target_state)

def start_simulation(circuit_file,
                     target_state=None,
                     run_cirq=True):
    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    rank = comm.Get_rank()
    if rank==0:
        circuit = read_circuit_file(circuit_file)
        comm.bcast(circuit)
    else:
        circuit=None
        circuit=comm.bcast(circuit,root=0)
    sim = Simulator()
    final_state_qtree = sim.simulate(
        circuit,
        parallel=True,
        save_graphs = False,
    )
    return sim.eval_time
    #print("---qtree_%i--- %s seconds ---" % (rank,time.time() - start_time))

    if rank==0:
        print("qtree", final_state_qtree.round(4))
        if run_cirq:
            cirq_sim = cirq.google.XmonSimulator()
            cirq_circuit = circuit.convert_to_cirq()
            start_time = time.time()
            cirq_result =cirq_sim.simulate(cirq_circuit)
            print("---cirq--- %s seconds ---" % (time.time() - start_time))
            # print(cirq_circuit)
            print("cirq ",cirq_result.final_state.round(4))

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
    #print(circuit.circuit)
    return circuit

if __name__=="__main__":
    main()
