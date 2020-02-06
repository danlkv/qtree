"""
Performance testing for Qtree calculations
"""
import numpy as np
import tensorflow as tf
import time
import pandas as pd
import subprocess
import cProfile

import qtree.operators as ops
import qtree.optimizer as opt
import qtree.graph_model as gm
import qtree.tf_framework as tffr
import qtree.np_framework as npfr
import qtree.utils as utils
import qtree.system_defs as defs

from qtree.logger_setup import log
from mpi4py import MPI
from matplotlib import pyplot as plt

from matplotlib.ticker import MaxNLocator

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

# controls default text sizes
plt.rc('font', size=SMALL_SIZE)
# fontsize of the axes title
plt.rc('axes', titlesize=BIGGER_SIZE)
# fontsize of the x and y labels
plt.rc('axes', labelsize=MEDIUM_SIZE)
# fontsize of the tick labels
plt.rc('xtick', labelsize=SMALL_SIZE)
# fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)
# legend fontsize
plt.rc('legend', fontsize=MEDIUM_SIZE)
# fontsize of the figure title
plt.rc('figure', titlesize=BIGGER_SIZE)


def profile_decorator(filename=None, comm=MPI.COMM_WORLD):
    def prof_decorator(f):
        def wrap_f(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()

            if filename is None:
                pr.print_stats()
            else:
                filename_r = filename + ".{}".format(comm.rank)
                pr.dump_stats(filename_r)

            return result
        return wrap_f
    return prof_decorator


def time_single_amplitude_tf(
        filename, target_state,
        quickbb_command=defs.QUICKBB_COMMAND):
    """
    Returns the time of a single amplitude evaluation.
    The circuit is loaded from the filename and the
    amplitude of the state is calculated. The time excludes
    file loading and quickbb operation.
    """
    n_qubits, buckets, free_vars = opt.read_buckets(filename)
    graph = opt.buckets2graph(buckets)

    # Calculate elimination order with QuickBB
    peo, treewidth = gm.get_peo(graph)

    # Transform graph to the elimination order
    graph_optimal, label_dict = gm.relabel_graph_nodes(
        graph, dict(zip(peo, range(1, len(peo) + 1)))
    )

    # Calculate costs
    mem_costs, flop_costs = gm.cost_estimator(graph_optimal)
    mem_max = np.sum(np.array(mem_costs, dtype=np.double))
    flop = np.sum(np.array(flop_costs, dtype=np.double))
    log.info('Evaluation cost:\n memory: {:e} flop: {:e}'.format(
        mem_max, flop))

    #@profile_decorator(filename='sequential_tf_cprof')
    def computational_core(buckets, peo):
        perm_buckets = opt.reorder_buckets(buckets, peo)

        tf_buckets, placeholder_dict = tffr.get_tf_buckets(
            perm_buckets, n_qubits)

        comput_graph = opt.bucket_elimination(
            tf_buckets, tffr.process_bucket_tf)

        feed_dict = tffr.assign_placeholder_values(
            placeholder_dict,
            target_state, n_qubits)
        amplitude = tffr.run_tf_session(comput_graph, feed_dict)
        return amplitude

    # Start measurement
    start_time = time.time()
    amplitude = computational_core(buckets, peo)
    end_time = time.time()

    return end_time - start_time, mem_max, flop, treewidth


def time_single_amplitude_np(
        filename, target_state,
        quickbb_command=defs.QUICKBB_COMMAND):
    """
    Returns the time of a single amplitude evaluation.
    The circuit is loaded from the filename and the
    amplitude of the state is calculated. The time excludes
    file loading and quickbb operation.
    """
    n_qubits, buckets, free_vars = opt.read_buckets(filename)
    graph = opt.buckets2graph(buckets)

    # Calculate elimination order with QuickBB
    peo, treewidth = gm.get_peo(graph)

    # Transform graph to the elimination order
    graph_optimal, label_dict = gm.relabel_graph_nodes(
        graph, dict(zip(peo, range(1, len(peo) + 1)))
    )

    # Calculate costs
    mem_costs, flop_costs = gm.cost_estimator(graph_optimal)
    mem_max = np.sum(np.array(mem_costs, dtype=np.double))
    flop = np.sum(np.array(flop_costs, dtype=np.double))
    log.info('Evaluation cost:\n memory: {:e} flop: {:e}'.format(
        mem_max, flop))

    #@profile_decorator(filename='sequential_np_cprof')
    def computational_core(buckets, peo):
        perm_buckets = opt.reorder_buckets(buckets, peo)

        np_buckets = npfr.get_np_buckets(
            perm_buckets, n_qubits, target_state)

        amplitude = opt.bucket_elimination(
            np_buckets, npfr.process_bucket_np)
        return amplitude

    # Start measurement
    start_time = time.time()
    amplitude = computational_core(buckets, peo)
    end_time = time.time()

    return end_time - start_time, mem_max, flop, treewidth


def time_single_amplitude_tf_mpi(
        filename, target_state,
        n_var_parallel_min=0,
        mem_constraint=defs.MAXIMAL_MEMORY,
        n_var_parallel_max=None,
        quickbb_command=defs.QUICKBB_COMMAND):
    """
    Returns the time of a single amplitude evaluation.
    The circuit is loaded from the filename and the
    amplitude of the state is calculated. The time excludes
    file loading and quickbb operation.
    """
    def prepare_mpi_environment(filename,
                                n_var_parallel_min,
                                mem_constraint,
                                n_var_parallel_max):
        n_qubits, buckets, free_vars = opt.read_buckets(filename)
        graph = opt.buckets2graph(buckets)

        # Split graphical model to parallelize
        (idx_parallel,
         reduced_graph) = gm.split_graph_with_mem_constraint_greedy(
            graph,
            n_var_parallel_min=n_var_parallel_min,
            mem_constraint=mem_constraint,
            n_var_parallel_max=n_var_parallel_max)

        # Calculate elimination order with QuickBB
        peo, treewidth = gm.get_peo(reduced_graph)

        # Transform graph to the elimination order
        graph_optimal, label_dict = gm.relabel_graph_nodes(
            reduced_graph,
            dict(zip(peo, range(1, len(peo) + 1)))
        )

        # Estimate cost
        mem_costs, flop_costs = gm.cost_estimator(graph_optimal)
        mem_max = np.sum(
            np.array(mem_costs, dtype=np.double)) * 2**len(idx_parallel)
        flop = np.sum(
            np.array(flop_costs, dtype=np.double)) * 2**len(idx_parallel)
        log.info('Evaluation cost:')
        log.info(' total:\n  memory: {:e} flop: {:e}'.format(
            mem_max, flop))
        log.info(' per node:\n  memory: {:e} flop: {:e}'.format(
            np.sum(mem_costs), np.sum(flop_costs)))

        # Permute buckets to the order of optimal contraction
        perm_buckets = opt.reorder_buckets(
            buckets, peo + idx_parallel)

        # Transform tensor labels in buckets to tensorflow placeholders
        # Reset Tensorflow graph as it may store
        # all tensors ever used before
        tf.reset_default_graph()
        tf_buckets, placeholder_dict = tffr.get_tf_buckets(
            perm_buckets, n_qubits)

        # Apply slicing as we parallelize over some variables
        sliced_tf_buckets, pdict_sliced = tffr.slice_tf_buckets(
            tf_buckets, placeholder_dict, idx_parallel)

        # Do symbolic computation of the result
        result = tf.identity(
            opt.bucket_elimination(
                sliced_tf_buckets, tffr.process_bucket_tf),
            name='result'
        )

        env = dict(
            n_qubits=n_qubits,
            idx_parallel=idx_parallel,
            input_names=list(pdict_sliced.keys()),
            tf_graph_def=tf.get_default_graph().as_graph_def(),
            costs=(mem_max, flop, treewidth, len(idx_parallel))
        )
        return env

    #@profile_decorator(filename='parallel_tf_cprof')
    def computational_core(env):

        # restore tensorflow graph, extract inputs and outputs
        tf.reset_default_graph()
        tf.import_graph_def(env['tf_graph_def'], name='')
        placeholder_dict = tffr.extract_placeholder_dict(
            tf.get_default_graph(),
            env['input_names']
        )
        result = tf.get_default_graph().get_tensor_by_name('result:0')

        # restore other parts of the environment
        n_qubits = env['n_qubits']
        idx_parallel = env['idx_parallel']

        feed_dict = tffr.assign_placeholder_values(
            placeholder_dict,
            target_state, n_qubits)

        amplitude = 0
        for slice_dict in utils.slice_values_generator(
                comm_size, rank, idx_parallel):
            parallel_vars_feed = {
                placeholder_dict[key]: val for key, val
                in slice_dict.items()}

            feed_dict.update(parallel_vars_feed)
            amplitude += tffr.run_tf_session(result, feed_dict)

        return amplitude

    comm = MPI.COMM_WORLD
    comm_size = comm.size
    rank = comm.rank

    if rank == 0:
        env = prepare_mpi_environment(filename,
                                      n_var_parallel_min,
                                      mem_constraint,
                                      n_var_parallel_max)
    else:
        env = None

    # Start time measurement
    start_time = time.time()

    # Synchronize processes
    env = comm.bcast(env, root=0)

    # Do main calculation
    amplitude = computational_core(env)

    # Collect results from all workers
    amplitude = comm.reduce(amplitude, op=MPI.SUM, root=0)

    end_time = time.time()
    elapsed_time = end_time - start_time

    comm.bcast(elapsed_time, root=0)

    mem_max, flop, treewidth, n_var_parallel = env['costs']

    return elapsed_time, mem_max, flop, treewidth, n_var_parallel


def time_single_amplitude_np_mpi(
        filename, target_state,
        n_var_parallel_min=0,
        mem_constraint=defs.MAXIMAL_MEMORY,
        n_var_parallel_max=None,
        quickbb_command=defs.QUICKBB_COMMAND):
    """
    Returns the time of a single amplitude evaluation.
    The circuit is loaded from the filename and the
    amplitude of the state is calculated. The time excludes
    file loading and quickbb operation.
    """
    def prepare_mpi_environment(filename,
                                n_var_parallel_min,
                                mem_constraint,
                                n_var_parallel_max):
        n_qubits, buckets, free_vars = opt.read_buckets(filename)
        graph = opt.buckets2graph(buckets)

        # Split the graph to parallelize
        (idx_parallel,
         reduced_graph) = gm.split_graph_with_mem_constraint_greedy(
            graph,
            n_var_parallel_min=n_var_parallel_min,
            mem_constraint=mem_constraint,
            n_var_parallel_max=n_var_parallel_max
        )

        # Calculate elimination order with QuickBB
        peo, treewidth = gm.get_peo(reduced_graph)

        # Transform graph to the elimination order
        graph_optimal, label_dict = gm.relabel_graph_nodes(
            reduced_graph,
            dict(zip(peo, range(1, len(peo) + 1)))
        )

        # Estimate cost
        mem_costs, flop_costs = gm.cost_estimator(graph_optimal)
        mem_max = np.sum(
            np.array(mem_costs, dtype=np.double)) * 2**len(idx_parallel)
        flop = np.sum(
            np.array(flop_costs, dtype=np.double)) * 2**len(idx_parallel)
        log.info('Evaluation cost:')
        log.info(' total:\n  memory: {:e} flop: {:e}'.format(
            mem_max, flop))
        log.info(' per node:\n  memory: {:e} flop: {:e}'.format(
            np.sum(mem_costs), np.sum(flop_costs)))

        # Permute buckets to the order of optimal contraction
        perm_buckets = opt.reorder_buckets(
            buckets, peo + idx_parallel)

        env = dict(
            n_qubits=n_qubits,
            idx_parallel=idx_parallel,
            buckets=perm_buckets,
            costs=(mem_max, flop, treewidth, len(idx_parallel))
        )
        return env

    #@profile_decorator(filename='parallel_np_cprof')
    def computational_core(env):

        # restore buckets
        buckets = env['buckets']

        # restore other parts of the environment
        n_qubits = env['n_qubits']
        idx_parallel = env['idx_parallel']

        # Transform label buckets to Numpy buckets
        np_buckets = npfr.get_np_buckets(
            buckets, n_qubits, target_state)

        amplitude = 0
        for slice_dict in utils.slice_values_generator(
                comm_size, rank, idx_parallel):
            # Slice Numpy buckets along the parallelized vars
            sliced_buckets = npfr.slice_np_buckets(
                np_buckets, slice_dict, idx_parallel)
            amplitude += opt.bucket_elimination(
                sliced_buckets, npfr.process_bucket_np)

        return amplitude

    comm = MPI.COMM_WORLD
    comm_size = comm.size
    rank = comm.rank

    if rank == 0:
        env = prepare_mpi_environment(filename,
                                      n_var_parallel_min,
                                      mem_constraint,
                                      n_var_parallel_max)
    else:
        env = None

    # Start time measurement
    start_time = time.time()

    # Synchronize processes
    env = comm.bcast(env, root=0)

    # Do main calculation
    amplitude = computational_core(env)

    # Collect results from all workers
    amplitude = comm.reduce(amplitude, op=MPI.SUM, root=0)

    end_time = time.time()
    elapsed_time = end_time - start_time

    comm.bcast(elapsed_time, root=0)

    mem_max, flop, treewidth, n_var_parallel = env['costs']

    return elapsed_time, mem_max, flop, treewidth, n_var_parallel


def collect_timings(
        out_filename,
        grid_sizes=[4, 5],
        depths=list(range(10, 15)),
        path_to_testcases='./test_circuits/inst/cz_v2',
        timing_fn=time_single_amplitude_tf):
    """
    Runs timings for test circuits with grid size equal to grid_sizes
    and outputs results to a pandas.DataFrame, and saves to out_filename
    (as pickle).
    timing_fn is a sequential (non-MPI) function
    """

    try:
        data = pd.read_pickle(out_filename)
    except FileNotFoundError:
        # lays down the structure of data
        data = pd.DataFrame(
            [],
            index=['exec_time', 'total_time',
                   'mem_max', 'flop', 'treewidth'],
            columns=pd.MultiIndex.from_product(
                [[], []], names=['grid size', 'depth']))

    total_tests = len(grid_sizes)*len(depths)
    log.info(f'Will run {total_tests} tests')

    # suffix of the test case file. Should we make it random?
    test_id = 2
    for n_grid, grid_size in enumerate(grid_sizes):
        log.info('Running grid = {}, [{}/{}]'.format(
            grid_size, n_grid+1, len(grid_sizes)))

        for n_depth, depth in enumerate(depths):
            log.info('Running depth = {}, [{}/{}]'.format(
                depth, n_depth+1, len(depths)))

            testfile = '/'.join((
                path_to_testcases,
                f'{grid_size}x{grid_size}',
                f'inst_{grid_size}x{grid_size}_{depth}_{test_id}.txt'
            ))

            # Will calculate "1111...1" target amplitude
            target_state = 2**(grid_size**2) - 1

            # Measure time and get predicted costs
            start_time = time.time()
            exec_time, *costs = timing_fn(
                testfile, target_state)
            end_time = time.time()
            total_time = end_time - start_time

            # Extract cost information
            mem_max, flop, treewidth = costs

            # Merge current result with the rest
            data[grid_size, depth] = [exec_time, total_time,
                                      mem_max, flop, treewidth]
            # Save result at every iteration
            data.to_pickle(out_filename)

    return data


def collect_timings_mpi(
        out_filename,
        grid_sizes=[4, 5],
        depths=list(range(10, 15)),
        path_to_testcases='./test_circuits/inst/cz_v2',
        timing_fn_mpi=time_single_amplitude_np_mpi,
        n_var_parallel_min=7,
        n_var_parallel_max=8):
    """
    Runs timings for test circuits with grid size equal to grid_sizes
    and outputs results to a pandas.DataFrame, and saves to out_filename
    (as pickle).
    This version supports execution by mpiexec. Running
    mpiexec -n 1 python <:py:meth:`collect_timings_mpi`>
    will produce different timings than :py:meth:`collect_timings`
    timing_fn_mpi should be MPI-friendly timing function
    """

    comm = MPI.COMM_WORLD
    comm_size = comm.size
    rank = comm.rank

    if rank == 0:
        try:
            data = pd.read_pickle(out_filename)
        except FileNotFoundError:
            # lays down the structure of data
            data = pd.DataFrame(
                [],
                index=['exec_time', 'total_time',
                       'mem_max', 'flop', 'treewidth', 'n_var_parallel'],
                columns=pd.MultiIndex.from_product(
                    [[], []],
                    names=['grid size', 'depth']))

        total_tests = len(grid_sizes)*len(depths)
        log.info(f'Will run {total_tests} tests')
        log.info(f'Will run {comm_size} paralell processes')
    else:
        data = None

    # suffix of the test case file. Should we make it random?
    test_id = 2
    for n_grid, grid_size in enumerate(grid_sizes):
        if rank == 0:
            log.info('Running grid = {}, [{}/{}]'.format(
                grid_size, n_grid+1, len(grid_sizes)))

        for n_depth, depth in enumerate(depths):
            if rank == 0:
                log.info('Running depth = {}, [{}/{}]'.format(
                    depth, n_depth+1, len(depths)))

            testfile = '/'.join((
                path_to_testcases,
                f'{grid_size}x{grid_size}',
                f'inst_{grid_size}x{grid_size}_{depth}_{test_id}.txt'
            ))

            # Will calculate "1111...1" target amplitude
            target_state = 2**(grid_size**2) - 1

            # Set the number of parallelized variables=5 (max 32 threads)
            # n_var_parallel = 5

            # Synchronize processes
            comm.bcast(testfile, root=0)
            comm.bcast(target_state, root=0)

            # Measure time
            start_time = time.time()
            exec_time, *costs = timing_fn_mpi(
                testfile, target_state, n_var_parallel_min=n_var_parallel_min,
                mem_constraint=defs.MAXIMAL_MEMORY,
                n_var_parallel_max=n_var_parallel_max)
            end_time = time.time()
            total_time = end_time - start_time
            mem_max, flop, treewidth, n_var_parallel = costs

            # Get maximal time as it determines overall time
            comm.reduce(exec_time, op=MPI.MAX, root=0)
            comm.reduce(total_time, op=MPI.MAX, root=0)

            if rank == 0:  # Parent process. Store results
                # Merge current result with the rest
                data[grid_size, depth] = [exec_time, total_time,
                                          mem_max, flop, treewidth,
                                          n_var_parallel]
                # Save result at each iteration
                data.to_pickle(out_filename)

    return data


def collect_timings_npar(
        testfile,
        vars_parallel,
        out_filename,
        timing_fn_mpi=time_single_amplitude_np_mpi):
    """
    Runs timings for a single test circuit with different number of parallel
    variables. This version supports execution by mpiexec.
    """

    comm = MPI.COMM_WORLD
    comm_size = comm.size
    rank = comm.rank

    if rank == 0:
        try:
            data = pd.read_pickle(out_filename)
        except FileNotFoundError:
            # lays down the structure of data
            data = pd.DataFrame(
                [],
                index=['exec_time', 'total_time',
                       'mem_max', 'flop', 'treewidth'],
                columns=pd.Index(
                    [],
                    name='n_var_parallel'))

        total_tests = len(vars_parallel)
        log.info(f'Will run {total_tests} tests')
        log.info(f'Will run {comm_size} paralell processes')
    else:
        data = None

    for n, n_var_parallel in enumerate(vars_parallel):
        if rank == 0:
            log.info('Running nparallel = {}, [{}/{}]'.format(
                n_var_parallel, n+1, total_tests))

        # Will calculate "1000..000" target amplitude
        target_state = 1

        # Synchronize processes
        comm.bcast(testfile, root=0)
        comm.bcast(target_state, root=0)

        # Measure time
        start_time = time.time()
        exec_time, *costs = timing_fn_mpi(
            testfile, target_state, n_var_parallel_min=n_var_parallel,
            mem_constraint=defs.MAXIMAL_MEMORY, n_var_parallel_max=n_var_parallel+1)
        end_time = time.time()
        total_time = end_time - start_time
        mem_max, flop, treewidth, _ = costs

        # Get maximal time as it determines overall time
        comm.reduce(exec_time, op=MPI.MAX, root=0)
        comm.reduce(total_time, op=MPI.MAX, root=0)

        if rank == 0:  # Parent process. Store results
            # Merge current result with the rest
            data[n_var_parallel] = [exec_time, total_time,
                                    mem_max, flop, treewidth]
            # Save result at each iteration
            data.to_pickle(out_filename)

    return data


def collect_timings_for_multiple_processes(
        filename_base='output/test', n_processes=[1], extra_args=[]):
    """
    Run :py:meth:`collect_timings_mpi` with different number of mpi processes

    Parameters
    ----------
    filename_base : str
           base of the output filename to be appended with process #
    n_processes : list, default [1]
           number of processes
    extra_args : list, default []
           additional arguments to :py:meth:`collect_timings_mpi`
    """
    for n_proc in n_processes:
        filename = filename_base + '_' + str(n_proc) + '.p'
        sh = "mpiexec -n {} -binding -bind-to:core ".format(n_proc)
        sh += "python -c 'from scratchpad.performance_test import collect_timings_mpi,time_single_amplitude_tf_mpi,time_single_amplitude_np_mpi;collect_timings_mpi(\"{}\",{})'".format(
            filename, ','.join(map(str, extra_args)))
        print(sh)

        process = subprocess.Popen(sh, shell=True)
        process.communicate()


def extract_parallel_efficiency(
        seq_filename, par_filename_base,
        n_processes=[1, 2], grid_size=4,
        depth=10, time_id='exec_time'):
    """
    Calculates parallel efficiency from collected data
    """
    seq_data = pd.read_pickle(seq_filename)
    seq_time = seq_data[(grid_size, depth)][time_id]

    par_times = []
    efficiencies = []
    for n_proc in n_processes:
        filename = par_filename_base + '_' + str(n_proc) + '.p'
        par_data = pd.read_pickle(filename)
        par_time = par_data[(grid_size, depth)][time_id]

        par_times.append(par_time)
        efficiencies.append(seq_time / (par_time * n_proc))

    return efficiencies, n_processes


def extract_record_vs_gridsize(
        filename, grid_sizes,
        depth=10, rec_id='exec_time'):
    """
    Extracts record vs grid size from the timings data file
    for a fixed depth
    """
    data = pd.read_pickle(filename)

    points = []
    for grid_size in grid_sizes:
        point = data[(grid_size, depth)][rec_id]
        points.append(point)

    return points, grid_sizes


def extract_record_vs_depth(
        filename, depths,
        grid_size=4, rec_id='exec_time'):
    """
    Extracts record vs depth from the timings data file
    for a fixed grid_size
    """
    data = pd.read_pickle(filename)

    times = []
    for depth in depths:
        time = data[(grid_size, depth)][rec_id]
        times.append(time)

    return times, depths


def extract_flops_per_sec_vs_depth(
        filename, depths,
        grid_size=4, time_id='exec_time'):
    """
    Extracts flops per second vs depth from the timings data file
    for a fixed grid_size
    """
    data = pd.read_pickle(filename)

    flops_per_sec = []
    for depth in depths:
        time = data[(grid_size, depth)][time_id]
        flop = data[(grid_size, depth)]['flop']
        flops_per_sec.append(flop / time)

    return flops_per_sec, depths


def extract_flops_per_sec_vs_nprocesses(
        par_filename_base,
        n_processes=[1, 2], grid_size=4,
        depth=10, time_id='exec_time'):
    """
    Calculates flops per second vs the number
    of parallel processes from collected data
    """

    flops_per_sec = []
    for n_proc in n_processes:
        filename = par_filename_base + '_' + str(n_proc) + '.p'
        data = pd.read_pickle(filename)
        time = data[(grid_size, depth)][time_id]
        flop = data[(grid_size, depth)]['flop']
        flops_per_sec.append(flop / time)

    return flops_per_sec, n_processes


def plot_time_vs_depth(filename,
                       fig_filename='time_vs_depth.png',
                       grid_sizes=[4, 5],
                       depths=range(10, 30),
                       interactive=False):
    """
    Plots time vs depth for some number of grid sizes
    Data is loaded from filename
    """
    if not interactive:
        plt.switch_backend('agg')

    # grid_sizes = [4]
    # depths = range(10, 20)

    # Create empty canvas
    fig, axes = plt.subplots(1, len(grid_sizes), sharey=True,
                             figsize=(6*len(grid_sizes), 6))

    for n, grid_size in enumerate(grid_sizes):
        time, depths_labels = extract_record_vs_depth(
            filename, depths, grid_size, rec_id='exec_time')
        axes[n].semilogy(depths_labels, time, 'b-', label='time')
        axes[n].set_xlabel(
            'depth of {}x{} circuit'.format(grid_size, grid_size))
        axes[n].set_ylabel('log(time in seconds)')
        axes[n].legend(loc='upper left')

        flop, depths_labels = extract_record_vs_depth(
            filename, depths, grid_size, rec_id='flop'
        )
        n_var_parallel, depths_labels = extract_record_vs_depth(
            filename, depths, grid_size, rec_id='n_var_parallel'
        )

        total_flop = [f * 2**nvar for f, nvar in zip(flop, n_var_parallel)]
        right_ax = axes[n].twinx()
        right_ax.semilogy(depths_labels, total_flop, 'r-', label='flop')
        right_ax.legend(loc='lower right')

    fig.suptitle('Evaluation time dependence on the depth of the circuit')

    if interactive:
        fig.show()

    fig.savefig(fig_filename)


def plot_time_vs_gridsize(filename,
                          fig_filename='time_vs_gridsize.png',
                          grid_sizes=[5],
                          depths=[20, 25],
                          interactive=False):
    """
    Plots time vs gridsize for some number of depths
    Data is loaded from filename
    """
    if not interactive:
        plt.switch_backend('agg')

    # grid_sizes = [4]
    # depths = range(10, 20)

    # Create empty canvas
    fig, axes = plt.subplots(1, len(depths), sharey=True,
                             figsize=(6*len(depths), 6))

    for n, depth in enumerate(depths):
        time, gridsize_labels = extract_record_vs_gridsize(
            filename, grid_sizes=grid_sizes,
            depth=depth, rec_id='exec_time')
        axes[n].semilogy(gridsize_labels, time, 'b-', label='time')
        axes[n].set_xlabel(
            'size of depth={} circuit'.format(depth))
        axes[n].set_ylabel('log(time in seconds)')
        axes[n].legend(loc='upper left')

        treewidth, depths_labels = extract_record_vs_gridsize(
            filename, grid_sizes, depth, rec_id='treewidth'
        )
        # n_var_parallel, depths_labels = extract_record_vs_gridsize(
        #     filename, grid_sizes, depth, rec_id='n_var_parallel'
        # )

        # total_flop = [f * 2**nvar for f, nvar in zip(flop, n_var_parallel)]
        right_ax = axes[n].twinx()
        right_ax.semilogy(
            depths_labels, treewidth, 'r-', label='treewidth')
        right_ax.legend(loc='lower right')

    fig.suptitle('Evaluation time dependence on the size of the circuit')

    if interactive:
        fig.show()

    fig.savefig(fig_filename)


def plot_par_vs_depth_multiple(
        seq_filename, par_filename_base,
        n_processes=[1, 2], fig_filename='time_vs_depth_multiple.png',
        interactive=False):
    """
    Plots time vs depth for sequential and multiple MPI
    runs
    """
    grid_size = 5
    depths = list(range(10, 21))

    if not interactive:
        plt.switch_backend('agg')

    # Create empty canvas
    fig, axes = plt.subplots(1, len(n_processes)+1, sharey=True,
                             figsize=(12, 6))

    filenames = [seq_filename] + [par_filename_base + '_' +
                                  str(n_proc) + '.p'
                                  for n_proc in n_processes]
    titles = ['Sequential'] + ['n = {}'.format(n_proc)
                               for n_proc in n_processes]

    for n, (filename, title) in enumerate(zip(filenames, titles)):
        time, depths = extract_record_vs_depth(
            filename, depths, grid_size)
        axes[n].semilogy(depths, time)
        axes[n].set_xlabel('depth')
        axes[n].set_ylabel('log(time in seconds)')
        axes[n].set_title(title)

    fig.suptitle('Evaluation time vs depth of the {}x{} circuit\n'.format(
        grid_size, grid_size)
    )

    if interactive:
        fig.show()

    fig.savefig(fig_filename)


def plot_par_efficiency(
        seq_filename, par_filename_base,
        n_processes=[1, 2], fig_filename='efficiency.png',
        interactive=False):
    """
    Plots parallel efficiency for a given set of processors
    """
    grid_size = 9
    depth = 14

    if not interactive:
        plt.switch_backend('agg')

    efficiency, n_proc = extract_parallel_efficiency(
        seq_filename, par_filename_base,
        n_processes, grid_size, depth,
    )

    # Create empty canvas
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.plot(n_proc, efficiency)
    ax.set_xlabel(
        'number of processes')
    ax.set_ylabel('Efficiency')
    ax.set_title('Efficiency of MPI parallel code\n' +
                 ' for {}x{} qubits, {} layers'.format(
                     grid_size, grid_size, depth))

    if interactive:
        fig.show()

    fig.savefig(fig_filename)


def plot_flops_per_sec_vs_depth(
        filename,
        fig_filename='fps_vs_depth.png',
        grid_sizes=[5, 6, 7],
        depths=range(10, 25),
        interactive=False):
    """
    Plots flops per second vs depth for some number of grid sizes
    Data is loaded from filename
    """
    if not interactive:
        plt.switch_backend('agg')

    # grid_sizes = [6, 7]
    # depths = range(10, 16)

    # Create empty canvas
    fig, axes = plt.subplots(1, len(grid_sizes), sharey=True,
                             figsize=(6*len(grid_sizes), 6))

    for n, grid_size in enumerate(grid_sizes):
        flops_per_sec, depths = extract_flops_per_sec_vs_depth(
            filename, depths, grid_size)
        axes[n].semilogy(depths, flops_per_sec)
        # axes[n].axhline(y=8.388*10**9, color='k', linestyle='-.')
        # axes[n].text(1, 8.5*10**9, 'zAXPY', color='k')
        axes[n].set_xlabel(
            'depth of {}x{} circuit'.format(grid_size, grid_size))
        axes[n].set_ylabel('flops per second')
    fig.suptitle('Flops per second vs depth of the circuit')

    if interactive:
        fig.show()

    fig.savefig(fig_filename)


def plot_time_vs_n_var_parallel(
        filename,
        fig_filename='time_vs_n_var_parallel.png',
        interactive=False):
    """
    Plots time as a function of the number of parallelized variables
    """
    if not interactive:
        plt.switch_backend('agg')

    # Create empty canvas
    fig, ax = plt.subplots(1, 1, sharey=True,
                           figsize=(6, 6))

    data = pd.read_pickle(filename)
    times = data.loc['exec_time', :].get_values()
    costs_per_task = data.loc['flop', :].get_values()
    n_vars = data.loc['exec_time', :].index.get_values()

    costs_total = [cost * 2**n_var_parallel for cost, n_var_parallel
                   in zip(costs_per_task, n_vars)]
    ax.semilogy(n_vars, times, ls='', color='b', marker='o', label='time')
    ax.set_xlabel(
        'Number of parallelized variables')
    ax.set_ylabel('time')
    ax.set_title('Time and cost as a function \n of the number of parallelized variables')
    ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.semilogy(n_vars, costs_total, ls='', color='r', marker='*', label='cost')

    ax2.set_ylabel('predicted cost')
    ax2.legend(loc='lower right')

    if interactive:
        fig.show()

    fig.savefig(fig_filename)


def plot_fps_vs_n_var_parallel(
        filename,
        fig_filename='fps_vs_n_var_parallel.png',
        interactive=False):
    """
    Plots time as a function of the number of parallelized variables
    """
    if not interactive:
        plt.switch_backend('agg')

    # Create empty canvas
    fig, ax = plt.subplots(1, 1, sharey=True,
                           figsize=(6, 6))

    data = pd.read_pickle(filename)
    times = data.loc['exec_time', :].get_values()
    costs_per_task = data.loc['flop', :].get_values()
    n_vars = data.loc['exec_time', :].index.get_values()

    costs_total = [cost * 2**n_var_parallel for cost, n_var_parallel
                   in zip(costs_per_task, n_vars)]
    ax.semilogy(n_vars, costs_total / times, 'b-', marker='o', label='fps')
    ax.set_xlabel(
        'Number of parallelized variables')
    ax.set_ylabel('flops per second')
    ax.set_title('Performance as a function of the number of parallelized variables')

    if interactive:
        fig.show()

    fig.savefig(fig_filename)


def plot_treewidth_vs_depth(
        filename,
        fig_filename='tw_vs_depth.eps',
        grid_sizes=[4, 5],
        depths=range(10, 30),
        interactive=False):
    """
    Plots treewidth vs depth for some number of grid sizes
    Data is loaded from filename
    """
    if not interactive:
        plt.switch_backend('agg')

    # Create empty canvas
    fig, axes = plt.subplots(1, len(grid_sizes), sharey=True,
                             figsize=(6*len(grid_sizes), 6))
    if type(axes) is not np.ndarray:
        axes = [axes]

    for n, grid_size in enumerate(grid_sizes):
        time, depths_labels = extract_record_vs_depth(
            filename, depths, grid_size, rec_id='treewidth')
        axes[n].plot(depths_labels, time, 'b-o', label='treewidth')
        axes[n].set_xlabel(
            'depth of {}x{} circuit'.format(grid_size, grid_size))
        axes[n].set_ylabel('treewidth')
        axes[n].xaxis.set_major_locator(
            MaxNLocator(integer=True))
        #axes[n].legend(loc='upper left')

    # fig.suptitle('Treewidth dependence on\n the depth of the circuit')

    if interactive:
        fig.show()

    fig.savefig(fig_filename)


def plot_treewidth_vs_gridsize(
        filename,
        fig_filename='tw_vs_gridsize.eps',
        grid_sizes=[5, 6],
        depths=[20, 30],
        interactive=False):
    """
    Plots treewidth vs depth for some number of grid sizes
    Data is loaded from filename
    """
    if not interactive:
        plt.switch_backend('agg')

    # Create empty canvas
    fig, axes = plt.subplots(1, len(depths), sharey=True,
                             figsize=(6*len(depths), 6))
    if type(axes) is not np.ndarray:
        axes = [axes]

    for n, depth in enumerate(depths):
        time, depths_labels = extract_record_vs_gridsize(
            filename, grid_sizes, depth, rec_id='treewidth')
        axes[n].plot(depths_labels, time, 'b-o', label='treewidth')
        axes[n].set_xlabel(
            'lateral size of {} layer circuit'.format(depth))
        axes[n].set_ylabel('treewidth')
        axes[n].xaxis.set_major_locator(
            MaxNLocator(integer=True))
        #axes[n].legend(loc='upper left')

    # fig.suptitle('Treewidth dependence on\n the depth of the circuit')

    if interactive:
        fig.show()

    fig.savefig(fig_filename)


if __name__ == "__main__":
    # collect_timings('test_np.p', [4, 5], list(range(10, 21)),
    #                 timing_fn=time_single_amplitude_np)
    # collect_timings_for_multiple_processes(
    #     'output/test_np', [1, 2, 4],
    #     extra_args=[[4, 5], list(range(10, 21)), 'timing_fn_mpi=time_single_amplitude_tf_mpi']
    # )

    # time_single_amplitude_tf(
    #     'test_circuits/inst/cz_v2/5x5/inst_5x5_18_2.txt', 0)

    # time_single_amplitude_np_mpi(
    #     'test_circuits/inst/cz_v2/5x5/inst_5x5_18_2.txt', 0,
    #     n_var_parallel=3)

    # plot_time_vs_depth('hachiko_np_7.p',
    #                    fig_filename='time_vs_depth_7.png',
    #                    grid_sizes=[7, 7],
    #                    depths=range(10, 26),
    #                    interactive=True)
    # plot_par_vs_depth_multiple(
    #     'output/test_np.p',
    #     'output/test_np',
    #     n_processes=[1, 2, 4, 8, 16, 24, 32],
    #     fig_filename='time_vs_depth_multiple_np_hachiko.png',
    #     interactive=True)

    # plot_par_efficiency('output/test_np_1.p', 'output/test_np',
    #                     n_processes=[1, 2, 4, 8, 24, 32],
    #                     fig_filename='efficiency_np_hachiko.png',
    #                     interactive=True)

    # plot_flops_per_sec_vs_depth('hachiko_np.p',
    #                             fig_filename='fps_vs_depth.eps',
    #                             grid_sizes=[5, 6, 7],
    #                             depths=range(10, 26))
    # collect_timings_npar(
    #     'test_circuits/inst/cz_v2/5x5/inst_5x5_18_2.txt',
    #     [1, 5, 10, 15, 20, 25],
    #     'nvar_np.p',
    #     timing_fn_mpi=time_single_amplitude_np_mpi)

    # plot_time_vs_gridsize(
    #     'hachiko_np.p',
    #     fig_filename='time_vs_gridsize.png',
    #     grid_sizes=[4, 5, 6, 7, 8, 9],
    #     depths=[10, 15, 20],
    #     interactive=False)

    # grid_size = 7
    # depth = 20
    # plot_time_vs_n_var_parallel(
    #     f'nvar_np_{grid_size}x{grid_size}_{depth}.p',
    #     fig_filename=f'time_vs_n_var_parallel_{grid_size}x{grid_size}_{depth}.png'
    # )
    plot_treewidth_vs_depth(
        'output/cost_estimate_0.p',
        grid_sizes=[8],
        depths=range(20, 40),
        interactive=True
    )
    plot_treewidth_vs_gridsize(
        'output/cost_estimate_0.p',
        grid_sizes=[4, 5, 6, 7, 8, 9, 10, 11, 12],
        depths=[25],
        interactive=True
    )