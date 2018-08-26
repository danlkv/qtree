import time
import numpy as np
import matplotlib.pyplot as plt
from main  import start_simulation

def main():
    parameters = {
        'depth': list(range(10,22))
    }
    times = []
    i = 0
    for d in parameters['depth']:
        start = time.time()
        eval_time = run(d)
        end = time.time()
        dur = end - start
        print('duration with %s depth:\t %s'%(d,dur))
        times.append(eval_time)
        i+=1
        plt.plot(parameters['depth'][:i],times)
        plt.savefig('speed_depth5.png')
        plt.close()
    print(times)
    plt.plot(parameters['depth'],times)
    plt.savefig('speed_depth5.png')

    datafolder = './data/'
    data = np.array(times)
    filename = datafolder+'eval_times_0'
    print(filename)
    np.save(filename,data)

def run(depth):
    size= 5
    id = 2
    s = '%ix%i'%(size,size)
    folder = './%s/'%s
    filename = 'inst_%s_%i_%i.txt'%(s,depth,id)
    print("RUNNING " ,filename)
    eval_time = start_simulation(folder+filename,run_cirq=False)
    return eval_time

if __name__=='__main__':
    main()
