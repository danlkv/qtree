import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpi4py import MPI
import time

from .Variable import Variable
from .Tensor import Tensor
from .quickbb_api import gen_cnf, run_quickbb
import logging
log = logging.getLogger('qtree')

class Expression:
    def __init__(self,graph_model_plot=None):
        """Set Expression parameters and
        set empty lists for tensors and vars.

        Parameters
        ----------
        graph_model_plot: str, default None
            a path to file where to save a plot of graphical model
        """

        self.graph_model_plot=graph_model_plot
        self._tensors = []
        self._variables = []
        self._graph = nx.Graph()

    def __iadd__(self,tensor):
        """Append a tensor to Expression and add it's Variables
        if they're not already in set. Also, __update_graph()
        """
        if not isinstance(tensor,Tensor):
            raise Exception("expected Tensor but got",
                            tensor)
        self._tensors.append(tensor)
        self._variables+=tensor.variables
        self._variables = list(set(self._variables))
        #print('addwd vars to exp',self._variables)
        self.__update_graph(tensor)
        return self
    def __update_graph(self,tensor):
        for v in tensor.variables:
            self._graph.add_node(v._id)
        if len(tensor.variables)==2:
            self._graph.add_edge(
                tensor.variables[0]._id,
                tensor.variables[1]._id
            )
        elif len(tensor.variables)>2:
            # TODO make it work for more than 2 variables
            raise Exception('found a tensor with more than 2 vars.')
    def set_tensors(self,tensors):
        """A wrapper of `__iadd__()` for adding list of tensors
        Parameters
        ----------
        tensors: list [Tensor]
        """
        for t in tensors:
            self += t

    def set_order_from_qbb(self,free_vars):
        """Runs QuickBB and sets order of elimination
        Mutates `self._graph`: removes free vars
        Creates a plot to file `self.graph_model_plot` if defined
        Creates a config file for quickbb and some output files
        """
        cnffile = 'quickbb.cnf'
        graph = self._graph
        graph.remove_nodes_from([
            v._id for v in free_vars])
        if self.graph_model_plot:
            plt.figure(figsize=(10,10))
            nx.draw(graph,with_labels=True)
            plt.savefig(self.graph_model_plot)
        gen_cnf(cnffile,graph)
        ordering = run_quickbb(cnffile)
        print("Ordering from QuickBB is",ordering)
        self.set_order(ordering)

    def set_order(self,order):
        """Set order of elimination.
        Every variable is assigned an index, then
        the elimination will be performed by ascending indexes

        Parameters
        ----------
        order: list of integers
            indexes of variables, representing the desired order
        """
        i = 0
        for id in order:
            # set index i for variable with id id
            for v in self._variables:
                if v._id==id:
                    v.idx = i
            i+=1
        self._variables = sorted(
            self._variables, key=lambda v: v.idx)
        self.ordering=order

    def fix_vars_for_parallel(self):
        comm = MPI.COMM_WORLD
        nproc = comm.Get_size()
        rank = comm.Get_rank()
        print("RANK",rank)
        for i in range(nproc):
            if rank==i:
                # Here we assume that every variable has 2 vals:0 and 1
                # TODO: Support arbitary variable space size
                # var count is minimum k so that 2^k>nproc
                var_count=_next_exp_of2(nproc)
                leaf_count = np.power(2,var_count)
                non_merged_leafs = nproc - ( leaf_count-nproc )
                if rank<non_merged_leafs:
                    val_str = bin(rank)[2:].zfill(var_count)
                    print('rank, vstr',rank,val_str)
                    values =[int(x) for x in val_str]
                else:
                    val_str = bin(rank+(rank-non_merged_leafs))[2:-1].zfill(var_count-1)
                    values = [int(x) for x in val_str]
                i,j=0,0
                while i< (len(values)):
                    v = self._variables[j]
                    if not v.fixed:
                        v.fix(values[i])
                        i+=1
                    j+=1
                log.info('process with id %i evaluates with vals %s',
                         rank,val_str)
                self.__paralleled_vars = [x for x in self._variables if x._id in self.ordering[:len(val_str)]]
                print('parver',self.__paralleled_vars)

    def get_var_id(self,i):
        """Get Variables with given integer id
        returns all vars with var._id == i

        Parameters
        ----------
        i: int
            id of variable
        Returns
        ----------
        res: list [Variable]
        """
        res = []
        for v in self._variables:
            if v._id==i:
                res.append(v)
        return res

    def parallel_evaluate(self):
        comm = MPI.COMM_WORLD
        nproc = comm.Get_size()
        rank = comm.Get_rank()
        log.info('Evaluating the expression: %s',str(self))
        if rank==0:
            free_vars = self.free_vars
            start_time = time.time()
            self.set_order_from_qbb(free_vars)
            print("qbb-- %s seconds --" % (time.time() - start_time))
            ordering = comm.bcast(self.ordering,root=0)
        else:
            ordering=None
            ordering = comm.bcast(ordering,root=0)
            self.set_order(ordering)
            print('rank got order',rank,ordering)

        start_time = time.time()
        log.info('Slicing by fixed vars %s',[x for x in self._variables if x.fixed])
        self.fix_vars_for_parallel()
        for t in self._tensors:
            t.slice_if_fixed()
        log.info('Expression is now:%s',str(self))
        # Iterate over only non-free vars
        vs = [v for v in self._variables if not v.fixed]
        res = self._variable_eliminate(vs)
        if rank==0:
            results = [res]
            for i in range(1,nproc):
                results.append(comm.recv(source=i,tag=42))
            print('results', results)
            res = 0
            for r in results:
                if len(r)>1:
                    x = 1
                    for t in r:
                        x = x * t._tensor
                else:
                    x=r[0]._tensor
                res+=x
            return [Tensor(res)]
        else:
            req = comm.send(res ,dest=0,tag=42)
        print("eval%s--- %s seconds --" % (rank,time.time() - start_time))


    def evaluate(self,free_vars=None,parallel=False):
        """Evaluate the Expression by summing over non-free vars
        Uses variable elimination algorithm.
        Mutates the instance: removes all variables except free
        and removes all tensors except the resulting one

        Parameters
        ----------
        free_vars: list [Variable]
            the resulting tensor will have these variables
        Returns
        ----------
        self._tensors : list [Tensor]
            If everything went OK, list contains one tensor
            with rank=len(free_vars)
            If not, warning is printed (Check if graph connected)
        """
        # self.variables are expexted to be sorted
        if not free_vars:
            free_vars = self.free_vars
        log.info('Evaluating the expression: %s',str(self))

        start_time = time.time()
        self.set_order_from_qbb(free_vars)
        print("qbb-- %s seconds --" % (time.time() - start_time))
        log.info('Slicing by fixed vars %s',[x for x in self._variables if x.fixed])
        for t in self._tensors:
            t.slice_if_fixed()
        log.info('Expression is now:%s',str(self))
        # Iterate over only non-free vars
        vs = [v for v in self._variables if not v.fixed]
        return self._variable_eliminate(vs)

    def _variable_eliminate(self,vs):
        for var in vs:
            log.debug('expr %s',self)
            tensors_of_var = [t for t in self._tensors
                              if var in t.variables]
            log.info('Eliminating %s \twith %i tensors, sum ranks:%i',
                     var,len(tensors_of_var),
                    sum([t.rank for t in tensors_of_var]))
            log.debug('tensors of var: \n%s',
                  tensors_of_var,
                 )
            tensor_of_var = tensors_of_var[0].merge_with(
                tensors_of_var[1:])
            tensor_of_var.diagonalize_if_dupl()
            new_t = tensor_of_var.sum_by(var)
            log.debug('tensor after sum:\n%s',new_t)
            new_expr_tensors = []
            for t in self._tensors:
                if t not in tensors_of_var:
                    new_expr_tensors.append(t)
                else:
                    del t
            self._tensors = new_expr_tensors
            self._tensors.append(new_t)
        return self._tensors

    def __repr__(self):
        return ' '.join([str(t) for t in self._tensors])

def _next_exp_of2(n):
    print('n',n)
    n-=1
    e = 0
    while n>=1:
        n/=2
        e+=1
    print('e',e)
    return e

