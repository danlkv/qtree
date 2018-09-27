from branchAndBound import getBB,performBB
import data
import networkx as nx
import argparse
def elim(g,v):
    vs = list(g.neighbors(v))
    edges = [(vs[i],vs[j]) for i in range(len(vs)) for j in range(i)]
    g.remove_node(v)
    g.add_edges_from(edges)
    return g

def min_degreeV(g,nodes):
    # TODO: can do this faster
    deg = [(v,d) for v,d in g.degree if v in nodes]
    min_deg,vertex = 10000,0
    #print(deg)
    for v,d in deg:
        if d<min_deg:
            # TODO: if min deg ==1 return
            min_deg=d
            vertex=v
    if min_deg<1:
        print(v,nodes,deg)
    return vertex,min_deg

def mmw(g):
    lb = 0
    iter = 0
    while True:
        iter+=1
        nodes = list(g.nodes())
        if len(nodes)<2:
            return lb
        # todo: huge room for optimization here
        #---- Find min degree vertex
        vertex,degree = min_degreeV(g,nodes)
        #---- Find min degree v from his neighbors
        neig = list(g.neighbors(vertex))
        neig = [ v for v in neig if v!=vertex ]
        vertex2,_= min_degreeV(g,neig)
        #print(lb,iter,vertex,vertex2)
        # Self loops=False costed me 1.5 hours!
        g = nx.contracted_edge(g,(vertex,vertex2),self_loops=False)
        #data._save_graph(g,'gt%i.png'%iter)
        nodes = list(g.nodes())
        lb = max(lb,degree)

def minwidth(g):
    order = []
    deg = 0
    while True:
        nodes = list(g.nodes())
        if len(nodes)<2:
            return deg,order
        vertex,degree = min_degreeV(g,nodes)
        order.append(vertex)
        #g.remove_node(vertex)
        elim(g,vertex)
        deg = max(deg,degree)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('circuitfile', help='file with circuit')
    args = parser.parse_args()

    #graph = data.read_graph(args.circuitfile,max_depth=5)
    graph = nx.readwrite.from_graph6_bytes(b'Ss`bB???gD?BEE@@?K?B?E?K?@ooA?GGK')
    #return mmw(graph)
    data._save_graph(graph,'gt.png')
    def orderingWidth(s):
        #print('wait')
        graph_ = graph.copy()
        w = 0
        for v in s[1]:
            degree = len(list(graph_.neighbors(v)))
            w = max(w,degree)
            graph_ = elim(graph_,v)
        #d = list(graph.degree)
        #degrees = [ d for v,d in d if v in s[1]]
        # max(degrees)
        return w

    inital_state = (graph.copy(),[])
    def ub_fun(state,ub,lb):
        # This just terminates when path has all nodes
        if len(state[0].nodes())<2:
            print("graph eliminated")
            print("ub lb",ub,lb)
            print(state[1])
            ub = min(ub,lb)
        return ub

    def lb_fun(state,ub,lb):
        g = orderingWidth(state)
        h = mmw(state[0])
        f = max(g,h)
        if len(state[0].nodes())<3:
            print('width,mmw,lb',g,h,f)
        lb = f
        return lb

    def next_state(s):
        graph = s[0]

        for v in sorted(graph.nodes()):
            # --- Th 6.1
            def th61(s):
                return True
                if len(s[1])>0:
                    if v>s[1][-1]:
                        return True
                else: return True
            if th61(s):
                g_ = elim(s[0].copy(),v)
                x_ = s[1].copy()
                x_.append(v)
                ns = (g_,x_)
                yield ns

    bb_fun = getBB(ub_fun,lb_fun,next_state)
    performBB(bb_fun,
            next_state,
            inital_state
            )
if __name__=="__main__":
    main()

