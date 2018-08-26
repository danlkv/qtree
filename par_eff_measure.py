import numpy as np
import matplotlib.pyplot as plt

datafolder = './data/'
eff_data = []
start_nproc,end_nproc = 2,6
number = end_nproc-start_nproc +2

filename = datafolder+'eval_times_0.npy'
fig,axs = plt.subplots(1,1+end_nproc-start_nproc,
                       figsize=(25,5),sharey=True)
times0 = np.load(filename)

cmap = plt.get_cmap('jet')

axs[0].plot(np.log(times0))
axs[0].set_title('without parallelisation')
axs[0].set_ylabel('log(time in seconds)')
for i in range(start_nproc,end_nproc):
    filename = datafolder+'eval_times_%i.npy'%i
    times = np.load(filename)
    print('loaded times for %i nproc'%i)
    print(times0[-1],times[0,-1],i)
    colors = [cmap(i) for i in np.linspace(0, 1, len(times))]
    ax = axs[1+i-start_nproc]
    for j,p in enumerate(times):
        ax.plot(np.log(p),'*',dashes=[2,1],markersize='4',c=colors[j])
    eff_data.append( times0[-1]/(times[0,-1]*i))
    ax.set_title('%i processes'%(i))

fig.savefig('plots.png')
plt.close()
plt.xlabel('number of processes')
plt.ylabel('efficiency')
plt.plot( range(start_nproc,end_nproc), eff_data)
plt.savefig('efficency_plot.png')


