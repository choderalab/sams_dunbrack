from openmmtools.multistate import SAMSSampler, MultiStateReporter, MultiStateSamplerAnalyzer
from netCDF4 import Dataset
import mdtraj as md
from openmmtools import states
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

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
    positions = ncfile.variables['positions'][:,replica_index,:,:] # gets all frames from the first replica
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
    iteration = 250000
    reporter = MultiStateReporter('traj.nc', open_mode='r',checkpoint_interval=1)
    analyzer = MultiStateSamplerAnalyzer(reporter)
    Deltaf_ij, dDeltaf_ij = analyzer.get_free_energy()
    print(Deltaf_ij)
    print(dDeltaf_ij)
    return
     
# print traj
print_traj()
# analyze logZ using MBAR
analyze_logZ()
