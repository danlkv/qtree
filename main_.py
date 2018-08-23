import argparse,re
from logging import get_logger
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
    cnffile = 'quickbb.cnf'
    gen_cnf(cnffile,graph)
    outp = run_quickbb(cnffile)
    ordering = _get_ordering(str(outp))

    expr.set_order(ordering)
    tensors = expr.evaluate()
    t_res = tensors[0].reorder_by_id([0]+list(range(-1,-n_qubits-1,-1)))
    #print(t_res._tensor.round(4))

    #amp = naive_eliminate(graph,tensors)
    res_vec = tensor2vec(t_res._tensor[0])
    f = 200
    t = 230
    print('me  ',str(np.array(res_vec).round(4)))
    print('me  ',len(res_vec))
    print('cirq',state.round(4))
    log.info('amp of |0> is'+str(amp))
    #log.info("from cirq:"+str(target_amp))
    print()
def _get_ordering(out):
    m = re.search(r'(?P<peo>(\d+ )+).*Treewidth=(?P<treewidth>\s\d+)',
                  out, flags=re.MULTILINE | re.DOTALL )

    peo = [int(ii) for ii in m['peo'].split()]
    treewidth = int(m['treewidth'])
    return peo

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
