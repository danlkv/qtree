import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from .Variable import Variable
from .Tensor import Tensor
from .quickbb_api import gen_cnf, run_quickbb
import logging
log = logging.getLogger('qtree')

class Expression:
    def __init__(self,graph_model_plot=None):
        self.graph_model_plot=graph_model_plot
        self._tensors = []
        self._variables = []
        self._graph = nx.Graph()

    def __iadd__(self,tensor):
        if not isinstance(tensor,Tensor):
            raise Exception("expected Tensor but got",
                            tensor)
        self._tensors.append(tensor)
        self._variables+=tensor.variables
        self._variables = list(set(self._variables))
        #print('addwd vars to exp',self._variables)
        self.__update_graph(tensor)
        return self
    def set_tensors(self,tensors):
        for t in tensors:
            self += t
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
    def set_order_from_qbb(self,free_vars):
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
        """ Set index for every variable for ordering
        :param order: list of integers - ids of variables
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
    def get_var_id(self,i):
        res = []
        for v in self._variables:
            if v._id==i:
                res.append(v)
        return res

    def evaluate(self,free_vars=None):
        """ Evaluate the Expression by variable elimination
        algorithm
        :param free_vars: list of Variables - free vars
        :return: a Tensor with order=len(free_vars)
        """
        # variables are expexted to be sorted
        if not free_vars:
            free_vars = self.free_vars
        # Iterate over only non-free vars

        vs = []
        log.info('Evaluating the expression: %s',str(self))
        self.set_order_from_qbb(free_vars)
        for v in self._variables:
            if v not in free_vars:
                vs.append(v)
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
        return '.'.join([str(t) for t in self._tensors])
