import time
import matplotlib.pyplot as plt
from main_parallel  import start_simulation
from mpi4py import MPI
import numpy as np

def main():
    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    rank = comm.Get_rank()
    parameters = {
        'depth': list(range(10,22)),
        'size':[4,5],
    }
    subplots_count = len(parameters['size'])
    fig,axs = plt.subplots(1,subplots_count,sharey=True,
                 figsize=(12,6))
    for j,size in enumerate(parameters['size']):
        i = 0
        times = []
        for d in parameters['depth']:
            eval_time = run(d,size)
            print('duration %i with %s depth:\t %s'%
                  (rank,d,eval_time))
            times.append(eval_time)
            i+=1
            if rank==0:
                plt.plot(parameters['depth'][:i],times)
                plt.savefig('speed_depth5.png')
                plt.close()
        times = comm.gather(times,root=0)
        if rank==0:
            print("______******_____")
            print(times)
            print("SIZE %i ENDED"%size)
            for p in times:
                axs[j].plot(parameters['depth'],
                            [np.log(x) for x in p])
                fig.savefig('speed_depth_parallel_5.png')
                axs[j].set_xlabel(
                    'depth of %ix%i circuit'%(size,size))
                axs[j].set_ylabel('log(time in seconds)')

        fig.savefig('speed_depth_parallel_5.png')
        datafolder = './data/'
        data = np.array(times)
        filename = datafolder+'eval_times_%i'%nproc
        print(filename)
        np.save(filename,data)

def run(depth,size):
    id = 2
    s = '%ix%i'%(size,size)
    folder = './%s/'%s
    filename = 'inst_%s_%i_%i.txt'%(s,depth,id)
    print("RUNNING " ,filename)
    eval_time = start_simulation(folder+filename,run_cirq=False)
    return eval_time

if __name__=='__main__':
    main()
