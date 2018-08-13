#!/usr/bin/python3.6
from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph import GlobbingFilter
from pycallgraph.output import GraphvizOutput

from main import main

config = Config()
config.trace_filter = GlobbingFilter(include=[
        'main.*',
        'src.*',
        'qtree_numpy.*',
])


graphviz = GraphvizOutput(
    output_file='numpy_algo_v1.png')

with PyCallGraph(output=graphviz,config=config):
    main()
