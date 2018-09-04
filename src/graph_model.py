"""
Operations with graphical models
"""

import numpy as np
import re
import copy
import networkx as nx
from src.quickbb_api import gen_cnf, run_quickbb
from src.logger_setup import log


def relabel_graph_nodes(graph, label_dict=None):
    """
    Relabel graph nodes to consequtive numbers. If label
    dictionary is not provided, a relabelled graph and a
    dict {new : old} will be returned. Otherwise, the graph
    is relabelled (and returned) according to the label
    dictionary and an inverted dictionary is returned.

    Parameters
    ----------
    graph : networkx.Graph
            graph to relabel
    label_dict : optional, dict-like
            dictionary for relabelling {old : new}

    Returns
    -------
    new_graph : networkx.Graph
            relabeled graph
    label_dict : dict
            {new : old} dictionary for inverse relabeling
    """
    if label_dict is None:
        label_dict = {old: num for num, old in
                      enumerate(graph.nodes(data=False), 1)}
        new_graph = nx.relabel_nodes(graph, label_dict, copy=True)
    else:
        new_graph = nx.relabel_nodes(graph, label_dict, copy=True)

    # invert the dictionary
    label_dict = {val: key for key, val in label_dict.items()}

    return new_graph, label_dict


def get_peo(old_graph):
    """
    Calculates the elimination order for an undirected
    graphical model of the circuit. Optionally finds `n_qubit_parralel`
    qubits and splits the contraction over their values, such
    that the resulting contraction is lowest possible cost.
    Optionally fixes the values border nodes to calculate
    full state vector.

    Parameters
    ----------
    graph : networkx.Graph
            graph of the undirected graphical model to decompose

    Returns
    -------
    peo : list
          list containing indices in loptimal order of elimination
    max_mem : int
          leading memory order needed to perform the contraction (in floats)
    """

    cnffile = 'output/quickbb.cnf'
    initial_indices = old_graph.nodes()
    graph, label_dict = relabel_graph_nodes(old_graph)

    gen_cnf(cnffile, graph)
    out_bytes = run_quickbb(cnffile, './quickbb/run_quickbb_64.sh')

    # Extract order
    m = re.search(b'(?P<peo>(\d+ )+).*Treewidth=(?P<treewidth>\s\d+)',
                  out_bytes, flags=re.MULTILINE | re.DOTALL)

    peo = [int(ii) for ii in m['peo'].split()]

    # Map peo back to original indices
    peo = [label_dict[pp] for pp in peo]

    # find the rest of indices which quickBB did not spit out.
    # Those include isolated nodes (don't affect
    # scaling and may be added to the end of the variables list)
    # and something else

    isolated_nodes = nx.isolates(old_graph)
    peo = peo + sorted(isolated_nodes)

    # assert(set(initial_indices) - set(peo) == set())
    missing_indices = set(initial_indices)-set(peo)
    # The next line needs review. Why quickBB misses some indices?
    # It is here to make program work, but is it an optimal order?
    peo = peo + sorted(list(missing_indices))

    assert(sorted(peo) == sorted(initial_indices))
    log.info('Final peo from quickBB:\n{}'.format(peo))

    treewidth = int(m['treewidth'])

    return peo, 2**treewidth


def get_peo_parallel_random(old_graph, n_var_parallel=0):
    """
    Same as :py:meth:`get_peo`, but with randomly chosen nodes
    to parallelize over.

    Parameters
    ----------
    old_graph : networkx.Graph
                graph to contract (after eliminating variables which
                are parallelized over)
    n_var_parallel : int
                number of variables to eliminate by parallelization

    Returns
    -------
    peo : list
          list containing indices in loptimal order of elimination
    max_mem : int
          leading memory order needed to perform the contraction (in floats)
    idx_parallel : list
          variables removed by parallelization
    graph : networkx.Graph
          new graph without parallelized variables
    """
    graph = copy.deepcopy(old_graph)

    indices = list(graph.nodes())
    idx_parallel = np.random.choice(
        indices, size=n_var_parallel, replace=False)

    for idx in idx_parallel:
        graph.remove_node(idx)

    log.info("Removed indices by parallelization:\n{}".format(idx_parallel))

    peo, max_mem = get_peo(graph)

    return peo, max_mem, sorted(idx_parallel), graph


def get_peo_parallel_degree(old_graph, n_var_parallel=0):
    """
    Parallel-splitted version of :py:meth:`get_peo` with greedy
    choice of nodes to split.

    Parameters
    ----------
    old_graph : networkx.Graph
                graph to contract (after eliminating variables which
                are parallelized over)
    n_var_parallel : int
                number of variables to eliminate by parallelization

    Returns
    -------
    peo : list
          list containing indices in loptimal order of elimination
    max_mem : int
          leading memory order needed to perform the contraction (in floats)
    idx_parallel : list
          variables removed by parallelization
    graph : networkx.Graph
          new graph without parallelized variables
    """
    graph = copy.deepcopy(old_graph)

    # get nodes by degree in descending order
    nodes_by_degree = list((node, degree) for
                       node, degree in graph.degree())
    nodes_by_degree.sort(key=lambda pair: pair[1], reverse=True)

    idx_parallel = []
    for ii in range(n_var_parallel):
        node, degree = nodes_by_degree[ii]
        idx_parallel.append(node)

    for idx in idx_parallel:
        graph.remove_node(idx)

    log.info("Removed indices by parallelization:\n{}".format(idx_parallel))

    peo, max_mem = get_peo(graph)

    return peo, max_mem, sorted(idx_parallel), graph