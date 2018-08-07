import argparse
from src.logging import get_logger
get_logger()

from src.operators import *
from src.optimizer  import *
from src.quickbb_api import *
from cirq_test import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('circuitfile', help='file with circuit')
    parser.add_argument('target_state',
                        help='state x against which amplitude is computed')
    args = parser.parse_args()

    target_amp,state = get_amplitude_from_cirq(
       args.circuitfile, args.target_state)

    n_qubits, circuit = read_circuit_file(args.circuitfile)

    graph , expr = circ2graph(circuit)
    tensors = expr.evaluate()
    t_res = tensors[0].reorder_by_id([0,-1,-2,-3,-4])
    print(t_res._tensor.round(4))

    #amp = naive_eliminate(graph,tensors)
    res_vec = tensor2vec(t_res._tensor[0])
    print('me  ',str(np.array(res_vec).round(2)))
    print('cirq',state.round(2))
    log.info('amp of |0> is'+str(amp))
    #log.info("from cirq:"+str(target_amp))
    print()
    cnffile = 'quickbb.cnf'
    #gen_cnf(cnffile,graph)
    #run_quickbb(cnffile)

def tensor2vec(tensor):
    vec_len = 1
    for d in  tensor.shape:
        vec_len *= d
    index_len = len(tensor.shape)
    vec = []
    idx = index_len-1
    idx_vec = [0]*index_len
    def get_subtensor_vec(tensor,vec,result):
        if isinstance(tensor,np.ndarray):
            for i in range(2):
                get_subtensor_vec(tensor[i],vec,result)
        else:
            result.append(tensor)
    res = []
    get_subtensor_vec(tensor,idx_vec,res)
    return res

def get_by_idx_list(tensor,l):
    if isinstance(tensor,np.ndarray):
        if len(tensor.shape)==len(l):
            return get_by_idx_list(tensor[l[0]],l[1:])
        else:
            raise Exception('wrong number of indexes',tensor.shape,l)
    else:
        return tensor

if __name__=="__main__":
    main()
