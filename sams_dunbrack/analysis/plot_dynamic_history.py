from netCDF4 import Dataset
import mdtraj as md
from openmmtools import states
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

def plot_history():
    # get state, log weights and gamma history from traj.nc
    ncfile = Dataset('traj.nc', 'r')
    states = ncfile.variables['states']
    weights = ncfile.groups['online_analysis'].variables['log_weights_history']
    gammas = ncfile.groups['online_analysis'].variables['gamma_history']

    print(len(states))
    print(len(weights))
    print(len(gammas))

    '''    
    # read in the trajectory of cpptraj clusters from e2_cluster.out
    cpp_cluster = open('./cpptraj_cluster/e2-cluster.out','r')
    clusters = list()
    lines = cpp_cluster.readlines()
    for line in lines:
        if '#Frame' not in line:
            line = line.strip('\n')
            linesplit = line.split()
            clusters.append(int(linesplit[1]))
    '''
    assignment = [7, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 7, 7, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 2, 7, 7, 6, 7, 7, 6, 6, 7, 6, 7, 7, 6, 7, 6, 7, 6, 7, 0, 0, 0, 0, 0, 2, 0, 6, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 1, 5, 1, 1, 1, 5, 1, 5, 4, 1, 1, 4, 1, 1, 4, 5, 1, 1, 5, 1, 5, 5, 5, 5, 1, 1, 5, 5, 5, 1, 1, 1, 1, 1, 5, 4, 1, 1, 5, 1, 5, 5, 5, 4, 5, 5, 5, 0, 6, 5, 5, 1, 5, 5, 6, 1, 1, 1, 5, 1, 1, 6, 6, 5, 6, 6, 1, 1, 5, 6, 6, 5, 0, 5, 6, 6, 5, 5, 0, 5, 5, 5, 5, 5, 5, 5, 1, 5, 6, 5, 1, 0, 0, 5, 5, 5, 5, 1, 5, 1, 5, 6, 1, 6, 1, 5, 6, 5, 1, 6, 6, 5, 4, 1, 5, 1, 5, 5, 6, 5, 5, 1, 5, 1, 5, 5, 1, 1, 5, 5, 5, 1, 1, 5, 5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 6, 5, 0, 5, 1, 4, 1, 1, 1, 4, 1, 4, 1, 1, 4, 4, 1, 4, 4, 4, 4, 1, 4, 1, 1, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 4, 4, 1, 1, 6, 4, 4, 4, 6, 4, 6, 4, 4, 4, 4, 4, 6, 4, 6, 4, 1, 4, 4, 4, 1, 4, 1, 4, 4, 0, 4, 0, 1, 4, 4, 0, 4, 1, 1, 0, 0, 4, 4, 4, 0, 1, 4, 4, 4, 4, 4, 4, 4, 1, 1, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 6, 4, 4, 4, 4, 4, 4, 4, 6, 4, 4, 6, 1, 4, 4, 6, 4, 6, 7, 1, 1, 4, 6, 4, 1, 1, 1, 1, 4, 4, 1, 1, 1, 0, 1, 1, 4, 4, 1, 4, 1, 1, 4, 6, 6, 1, 4, 6, 4, 1, 1, 4, 4, 1, 1, 4, 6, 4, 0, 4, 4, 4, 0, 4, 1, 1, 4, 1, 4, 1, 4, 4, 6, 0, 1, 1, 1, 1, 1, 4, 4, 1, 1, 1, 4, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 1, 1, 4, 1, 4, 4, 1, 0, 4, 4, 4, 1, 1, 1, 1, 4, 1, 1, 4, 4, 4, 4, 4, 4, 4, 6, 4, 4, 6, 0, 4, 4, 4, 0, 4, 4, 4, 4, 6, 4, 1]
    #define colors
    c = dict()
    c[0] = '#FF0000' #red
    c[1] = '#FF8C00' #orange
    c[2] = '#FFD700' #yellow
    c[3] = '#32CD32' #green
    c[4] = '#48D1CC' #teal
    c[5] = '#0000FF' #blue
    c[6] = '#8A2BE2' #magenta
    c[7] = '#FF1493' #pink
    c[8] = '#393E46' #dark
    fig,ax = plt.subplots()
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    plt.xlim(0,500)
    ax1.set_ylim(-0.5,8.5)
    ax1.set_yticks([0,1,2,3,4,5,6,7,8])
    ax1.set_yticklabels([0,1,2,3,4,5,6,7,8], fontsize=8)
    ax2.set_ylim(-0.005, 0.115)
    ax2.set_yticks([0.000, 0.015, 0.030, 0.045, 0.060, 0.075, 0.090, 0.105])
    ax2.set_yticklabels([0.000, 0.015, 0.030, 0.045, 0.060, 0.075, 0.090, 0.105], fontsize=8)
    ax3.set_ylim(-80, 20)
    ax3.set_yticks([-60,-40,-20,0,20])
    ax3.set_yticklabels([-60,-40,-20,0,20], fontsize=8)
    ax4.set_ylim(-0.5, 7.5)
    ax4.set_yticks([0,1,2,3,4,5,6,7])
    ax4.set_xticks([0,100,200,300,400,500])
    ax4.set_yticklabels([0,1,2,3,4,5,6,7], fontsize=8)
    ax4.set_xticklabels([0,100,200,300,400,500], fontsize=8)
    for count in range(len(states)):
        if count % 500 == 0:
            ax1.scatter(int(count/500),states[count][0], color=c[5], s=4)
            ax2.scatter(int(count/500), gammas[count+1][0], color=c[4], s=4)
            for s in range(9):
                ax3.scatter(int(count/500), weights[count+1,s], color=c[s], s=4)
            ax4.scatter((int(count/500)), assignment[int(count/500)], color=c[1], s=4)
            plt.savefig(f'./frames/frame_{int(count/500)}.png',dpi=100)
            print(int(count/500))
    return
     
plot_history()
