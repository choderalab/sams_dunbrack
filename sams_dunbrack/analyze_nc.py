from netCDF4 import Dataset
import mdtraj as md
import numpy as np

def print_traj():
    experiment = 'both'
    pdbid = '3POZ'
    iteration = 150000
    # get some info directly from nc file
    ncfile = Dataset('traj_checkpoint.nc', 'r')

    # Get an MDTraj topology from a PDB file that matches the atom ordering in the saved trajectories
    print("Generating trajectory ...")
    mdtraj_topology = md.load(f'{pdbid}_chainA_minimized.pdb').topology
    # Now get the positions from the trajectory
    replica_index = 0
    positions = ncfile.variables['positions'][:,replica_index,:,:] # gets all frames from the first replica
    # Now create an MDTraj trajectory
    mdtraj_trajectory = md.Trajectory(positions, mdtraj_topology)
    # Write the trajectory in gromacs XTC format
    outfilename = f'{experiment}_{iteration}_traj.dcd'
    mdtraj_trajectory.save(outfilename)
    return

# print traj
print_traj()
