import networkx as nx
import re
import logging as log
import matplotlib.pyplot as plt

def read_graph(filename,max_depth=None):
    graph = nx.Graph()
    # perform the cirquit file processing
    print('reading file',filename)

    with open(filename,'r') as fp:
        qubit_count = int(fp.readline())
        log.info("There are {:d} qubits in circuit".format(qubit_count))
        n_ignored_layers = 0
        current_layer = 0
        current_var = qubit_count
        layer_variables = list(range(1, qubit_count+1))
        for line in fp:
            m = re.search(r'(?P<layer>[0-9]+) (?=[a-z])', line)
            if m is None:
                raise Exception("file format error at line {}".format(idx))
            # Read circuit layer by layer
            layer_num = int(m.group('layer'))

            if max_depth is not None and layer_num > max_depth:
                n_ignored_layers = layer_num - max_depth
                continue
            if layer_num > current_layer:
                current_layer = layer_num

            op_str = line[m.end():]
            m = re.search(
                    r'(?P<operation>h|t|cz|x_1_2|y_1_2) (?P<qubit1>[0-9]+) ?(?P<qubit2>[0-9]+)?',op_str)
            if m is None:
                raise Exception("file format error in {}".format(op_str))
            op_identif = m.group('operation')
            if m.group('qubit2') is not None:
                q_idx = int(m.group('qubit1')), int(m.group('qubit2'))
            else:
                q_idx = (int(m.group('qubit1')),)

            if op_identif=='cz':
                # cZ connects two variables with an edge
                var1 = layer_variables[q_idx[0]]
                var2 = layer_variables[q_idx[1]]
                graph.add_edge( var1, var2)
            elif op_identif=='h':
                pass
            elif op_identif=='t':
                # well, do nothing
                pass
            else:
                var1 = layer_variables[q_idx[0]]
                var2 = current_var+1
                graph.add_node(var2)
                graph.add_edge(var1, var2)
                current_var += 1
                layer_variables[q_idx[0]] = current_var
        if n_ignored_layers > 0:
            log.info("Ignored {} layers".format(n_ignored_layers))
        v = graph.number_of_nodes()
        e = graph.number_of_edges()
        print(f"Generated graph with {v} nodes and {e} edges")
        print(f"last index contains from {layer_variables}")

    return graph

def _save_graph(graph,path):
    plt.figure(figsize=(10,10))
    nx.draw_spectral(graph,
            node_color=(list(graph.nodes())),
            node_size=400,
            cmap=plt.cm.Blues,
            with_labels=True,
           )
    plt.savefig(path)
