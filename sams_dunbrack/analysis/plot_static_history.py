from netCDF4 import Dataset
import mdtraj as md
from openmmtools import states
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def print_traj():
    experiment = 'both'
    pdbid = '5UG9'
    iteration = 250000
    # get some info directly from nc file
    ncfile = Dataset('traj_checkpoint.nc', 'r')

    # Get an MDTraj topology from a PDB file that matches the atom ordering in the saved trajectories
    print("Generating trajectory ...")
    mdtraj_topology = md.load(f'{pdbid}_minimized.pdb').topology
    # Now get the positions from the trajectory
    replica_index = 0
    positions = ncfile.variables['positions'][1:2500001:250,replica_index,:,:] # gets all frames from the first replica
    # Now create an MDTraj trajectory
    mdtraj_trajectory = md.Trajectory(positions, mdtraj_topology)
    # Write the trajectory in gromacs XTC format
    outfilename = f'{experiment}_{iteration}_traj.dcd'
    mdtraj_trajectory.save(outfilename)
    return


# get info from multistatereporter
class MyComposableState(states.GlobalParameterState):
    switch = states.GlobalParameterState.GlobalParameter('switch', standard_value=1.0)
    phi0_dih0 = states.GlobalParameterState.GlobalParameter('phi0_dih0', standard_value=1.0)
    dphi_dih0 = states.GlobalParameterState.GlobalParameter('dphi_dih0', standard_value=1.0)
    phi0_dih1 = states.GlobalParameterState.GlobalParameter('phi0_dih1', standard_value=1.0)
    dphi_dih1 = states.GlobalParameterState.GlobalParameter('dphi_dih1', standard_value=1.0)
    phi0_dih2 = states.GlobalParameterState.GlobalParameter('phi0_dih2', standard_value=1.0)
    dphi_dih2 = states.GlobalParameterState.GlobalParameter('dphi_dih2', standard_value=1.0)
    phi0_dih3 = states.GlobalParameterState.GlobalParameter('phi0_dih3', standard_value=1.0)
    dphi_dih3 = states.GlobalParameterState.GlobalParameter('dphi_dih3', standard_value=1.0)
    phi0_dih4 = states.GlobalParameterState.GlobalParameter('phi0_dih4', standard_value=1.0)
    dphi_dih4 = states.GlobalParameterState.GlobalParameter('dphi_dih4', standard_value=1.0)
    phi0_dih5 = states.GlobalParameterState.GlobalParameter('phi0_dih5', standard_value=1.0)
    dphi_dih5 = states.GlobalParameterState.GlobalParameter('dphi_dih5', standard_value=1.0)
    phi0_dih6 = states.GlobalParameterState.GlobalParameter('phi0_dih6', standard_value=1.0)
    dphi_dih6 = states.GlobalParameterState.GlobalParameter('dphi_dih6', standard_value=1.0)
    phi0_dih7 = states.GlobalParameterState.GlobalParameter('phi0_dih7', standard_value=1.0)
    dphi_dih7 = states.GlobalParameterState.GlobalParameter('dphi_dih7', standard_value=1.0)
    phi0_dih8 = states.GlobalParameterState.GlobalParameter('phi0_dih8', standard_value=1.0)
    dphi_dih8 = states.GlobalParameterState.GlobalParameter('dphi_dih8', standard_value=1.0)
    phi0_dih9 = states.GlobalParameterState.GlobalParameter('phi0_dih9', standard_value=1.0)
    dphi_dih9 = states.GlobalParameterState.GlobalParameter('dphi_dih9', standard_value=1.0)
    r0_dis0 = states.GlobalParameterState.GlobalParameter('r0_dis0', standard_value=1.0)
    dr_dis0 = states.GlobalParameterState.GlobalParameter('dr_dis0', standard_value=1.0)
    r0_dis1 = states.GlobalParameterState.GlobalParameter('r0_dis1', standard_value=1.0)
    dr_dis1 = states.GlobalParameterState.GlobalParameter('dr_dis1', standard_value=1.0)
    r0_dis2 = states.GlobalParameterState.GlobalParameter('r0_dis2', standard_value=1.0)
    dr_dis2 = states.GlobalParameterState.GlobalParameter('dr_dis2', standard_value=1.0)
    r0_dis3 = states.GlobalParameterState.GlobalParameter('r0_dis3', standard_value=1.0)
    dr_dis3 = states.GlobalParameterState.GlobalParameter('dr_dis3', standard_value=1.0)
    r0_dis4 = states.GlobalParameterState.GlobalParameter('r0_dis4', standard_value=1.0)
    dr_dis4 = states.GlobalParameterState.GlobalParameter('dr_dis4', standard_value=1.0)
    r0_dis5 = states.GlobalParameterState.GlobalParameter('r0_dis5', standard_value=1.0)
    dr_dis5 = states.GlobalParameterState.GlobalParameter('dr_dis5', standard_value=1.0)
    r0_dis6 = states.GlobalParameterState.GlobalParameter('r0_dis6', standard_value=1.0)
    dr_dis6 = states.GlobalParameterState.GlobalParameter('dr_dis6', standard_value=1.0)
    r0_dis7 = states.GlobalParameterState.GlobalParameter('r0_dis7', standard_value=1.0)
    dr_dis7 = states.GlobalParameterState.GlobalParameter('dr_dis7', standard_value=1.0)
    r0_dis8 = states.GlobalParameterState.GlobalParameter('r0_dis8', standard_value=1.0)
    dr_dis8 = states.GlobalParameterState.GlobalParameter('dr_dis8', standard_value=1.0)
    r0_dis9 = states.GlobalParameterState.GlobalParameter('r0_dis9', standard_value=1.0)
    dr_dis9 = states.GlobalParameterState.GlobalParameter('dr_dis9', standard_value=1.0)
def analyze_logZ():
    from openmmtools.multistate import SAMSSampler, MultiStateReporter, MultiStateSamplerAnalyzer
    iteration = 250000
    reporter = MultiStateReporter('traj.nc', open_mode='r',checkpoint_interval=1)
    analyzer = MultiStateSamplerAnalyzer(reporter)
    Deltaf_ij, dDeltaf_ij = analyzer.get_free_energy()
    print(Deltaf_ij)
    print(dDeltaf_ij)
    return

def plot_history():
    # get state, log weights and gamma history from traj.nc
    ncfile = Dataset('traj.nc', 'r')
    y_state = list(); y_weight = dict(); y_gamma = list()
    states = ncfile.variables['states']
    weights = ncfile.groups['online_analysis'].variables['log_weights_history']
    gammas = ncfile.groups['online_analysis'].variables['gamma_history']
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
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    for s in range(9):
        count = 0
        x = list()
        y_weight[s] = list()
        for weight in weights[:,s]:
            if count % 250 == 0:
                x.append(count/250)
                y_weight[s].append(weight)
            count += 1
        ax3.plot(x, y_weight[s], color=c[s])
    count = 0
    for gamma in gammas:
        if count % 250 == 0:
            y_state.append(states[count])
            y_gamma.append(gamma[0])
        count += 1
    ax1.scatter(x,y_state, color=c[5])
    ax2.plot(x, y_gamma, color=c[4])
    plt.show()
    return
     
# print traj
#print_traj()
# analyze logZ using MBAR
#analyze_logZ()
plot_history()
