from netCDF4 import Dataset
import mdtraj as md
import numpy as np

def print_traj():
    experiment = 'state0'
    pdbid = '1M17'
    chain = 'A'
    iteration = '500ns'
    # get some info directly from nc file
    ncfile = Dataset('traj.nc', 'r')

    # Get an MDTraj topology from a PDB file that matches the atom ordering in the saved trajectories
    print("Generating trajectory ...")
    mdtraj_topology = md.load(f'{pdbid}_chain{chain}_minimized.pdb').topology
    # Now get the positions from the trajectory
    positions = ncfile.variables['coordinates'][:,:,:] # gets coordinates of all atoms from all frames
    # Now create an MDTraj trajectory
    mdtraj_trajectory = md.Trajectory(positions, mdtraj_topology)
    # Write the trajectory in gromacs XTC format
    outfilename = f'{experiment}_{iteration}_traj.dcd'
    mdtraj_trajectory.save(outfilename)
    return

# print traj
print_traj()
