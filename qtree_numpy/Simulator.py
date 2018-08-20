import networkx as nx
from .einsum import Expression, Tensor, Variable
import numpy as np
import logging
from .operators import cZ
log = logging.getLogger('qtree')

class Simulator:
    def __init__(self,
                 num_threads=1,
                 optimization='quickbb'):
        """
        Creates a Simulator instance
        :param num_threads: number of parallel jobs to use
        :param optimization: an algorithm that provides order
        for eliminaton
        """
        self.num_threads = num_threads
        self.optimization = optimization
    def simulate(self,circuit,inital_state=None,
                 graph_model_plot=None,
                 parallel=False
                ):
        """
        Simulate a cirquit
        :param circuit: a non-empty qtree.Circuit object
        :param inital_state: a numpy-array of length 2^N
        with complex amplitudes
        :param graph_model_plot: a file where a plot of graph model
        will be saved. Default is None
        :return: a Numpy-array of complex amplitudes for final state
        """
        self.__check_if_empty(circuit)
        qubit_count = circuit.qubit_count
        if not inital_state:
            # Set the |0> state by default
            inital_state = np.zeros(np.power(2,qubit_count),
                                    dtype=np.complex128)
            inital_state[0] = 1.+0j
        self.__check_inital_state_length(qubit_count,inital_state)

        e =  self.build_expr(circuit.circuit)
        e.graph_model_plot = graph_model_plot
        # the result is tensor of rank len(free_variables) = 1+N
        if parallel:
            tensors=e.parallel_evaluate()
        else:
            tensors = e.evaluate()
        if tensors:
            if len(tensors)>1:
                log.warn('something went wrong. make sure graph is connected')
                print(tensors)
            t_res = tensors[0].reorder_by_id([0]+list(range(-1,-qubit_count-1,-1)))
            # TODO: use custom qubit order
            return _tensor2vec(tensors[0]._tensor)

    def build_expr(self,circuit):
        """ builds an Expression out of list of layers [operator,...]
        :param curcuit: list of qOperation lists with first and last
        layer of hadamards on each
        """
        #circuit.reverse()

        # we start from 0 here to avoid problems with quickbb
        expr = Expression()
        vari = [Variable(0).fix(0)]
        free_vars = [vari[0]]

        # Process first layer, connect Variable(0) to N variables
        # works only if first and last layer consists of 
        # non diagonal op on every qubit
        first_layer = circuit[0]
        qubit_count = len(first_layer)
        for i in range(qubit_count):
            # create new variable
            vari.append(Variable(i+1))
            # Create new tensor and bind it with Variable(0)
            op = first_layer[i]
            tensor =Tensor(op.tensor)
            tensor.add_variables(vari[0],vari[i+1])
            tensor.name=op.name+'@'+str(op._qubits[0])
            expr+=tensor

        current_var = qubit_count
        variable_col= list(range(1,qubit_count+1))

        # Main circuit
        print('main')
        for layer in circuit[1:-1]:
            for op in layer:
                tensor = Tensor(op.tensor)
                variable_of_qubit = vari[variable_col[op._qubits[0]]]
                if not op.diagonal:
                    # Non-diagonal gate adds a new variable and
                    # an edge to graph
                    vari.append(Variable(current_var+1))
                    tensor.add_variables(
                        variable_of_qubit,
                        vari[current_var+1] )
                    current_var += 1

                    variable_col[op._qubits[0]] = current_var

                elif isinstance(op,cZ):
                    # cZ connects two variables with an edge
                    i1 = variable_col[op._qubits[0]]
                    i2 = variable_col[op._qubits[1]]
                    tensor.add_variables(vari[i1],vari[i2])
                else:
                    # just add variable corresponding to a qubit 
                    # operation acts on
                    tensor.add_variables(variable_of_qubit)
                tensor.name=op.name+'@'+str(op._qubits[0])
                expr+=tensor

        # Process last layer
        i = 1
        last_layer = circuit[-1]
        if len(last_layer)!=qubit_count:
            log.warn("Last layer should contain an operator on each qubit")
            print(last_layer)
        values_vars = [0,1,0,0]
        for op in last_layer:
            tensor = Tensor(op.tensor)
            # create a variable for each free qubit after the circuit
            xvar = Variable(-i)
            xvar.fix(values_vars[i-1])
            vari.append(xvar)
            free_vars.append(xvar)
            tensor.add_variables(vari[variable_col[i-1]],xvar)
            expr += tensor
            i+=1
        expr.free_vars = free_vars

        return expr

    def __check_if_empty(self,circuit):
        pass
    def __check_inital_state_length(self,qubit_count,state):
        if len(state)!=np.power(2,qubit_count):
            raise Exception("inital state length not equal 2^N")
        else:
            pass

def _tensor2vec(tensor):
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
    return np.array(res)

