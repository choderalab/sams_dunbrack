import os
import simtk.openmm as mm
from simtk.openmm import unit, version, Context, MonteCarloBarostat 
from simtk.openmm.app import PDBFile, PME, Simulation, StateDataReporter, HBonds, AmberPrmtopFile, AmberInpcrdFile
from openmmtools import states, mcmc, multistate
from features import featurize
import numpy as np
import tempfile
import yank
import logging

# set up basic parameters
experiment = 'wt_holo_both_4fs' # setting of the experiment (e.g. different combinations of the CVs)
pdbid = '2HYY' # PDB ID of the system
chain = 'A'
iteration = 250000
work_dir = f'/data/chodera/jiayeguo/projects/cv_selection/sams_simulation/new_trials/{pdbid}_{experiment}_{iteration}'
temperature = 310.15 * unit.kelvin
pressure = 1.0 * unit.atmospheres
ndihedrals = 7 # number of dihedrals we want to restrain
ndistances = 2 # number of distances we want to restrain
targets = list(range(8)) # list of dunbrack clusters (sams states) to bias to
coefficient = 1.0 # coefficient for force constant

# load prm or crd files of the minimized and equilibrated protein 
prmtop = AmberPrmtopFile('./complex_prep/02.ante.tleap/2HYY.com.wat.leap.prmtop')
pdb = PDBFile(f'{pdbid}_chain{chain}_holo_minequi.pdb')

print("OpenMM version:", version.version)
# use heavy hydrogens and constrain all hygrogen atom-involved bonds
system = prmtop.createSystem(nonbondedMethod=PME, rigidWater=True, nonbondedCutoff=1*unit.nanometer, hydrogenMass=4*unit.amu, constraints = HBonds)
system.addForce(MonteCarloBarostat(pressure, temperature))
# Encode the Dunbrack data (Modi and Dunbrack Jr., 2019)
import pandas as pd
from scipy import stats
df = pd.read_csv('./dunbrack_data_clean.csv')

data_mean = df.groupby('Cluster').aggregate({'X_Phi' : lambda x : stats.circmean(x, high=180.0, low=-180.0), 'X_Psi' : lambda x : stats.circmean(x, high=180.0, low=-180.0), 'Asp_Phi' : lambda x : stats.circmean(x, high=180.0, low=-180.0), 'Asp_Psi' : lambda x : stats.circmean(x, high=180.0, low=-180.0), 'Phe_Phi' : lambda x : stats.circmean(x, high=180.0, low=-180.0), 'Phe_Psi' : lambda x : stats.circmean(x, high=180.0, low=-180.0), 'Phe_Chi1' : lambda x : stats.circmean(x, high=180.0, low=-180.0), 'D1' : 'mean', 'D2' : 'mean'})

data_std = df.groupby('Cluster').aggregate({'X_Phi' : lambda x : stats.circstd(x, high=180.0, low=-180.0), 'X_Psi' : lambda x : stats.circstd(x, high=180.0, low=-180.0), 'Asp_Phi' : lambda x : stats.circstd(x, high=180.0, low=-180.0), 'Asp_Psi' : lambda x : stats.circstd(x, high=180.0, low=-180.0), 'Phe_Phi' : lambda x : stats.circstd(x, high=180.0, low=-180.0), 'Phe_Psi' : lambda x : stats.circstd(x, high=180.0, low=-180.0), 'Phe_Chi1' : lambda x : stats.circstd(x, high=180.0, low=-180.0), 'D1' : 'std', 'D2' : 'std'})
print(data_mean)
print(data_std)
dunbrack_mean = dict(); dunbrack_std = dict()

# convert cluster names to integers
names = dict()
names[0] = 'BLAminus'
names[1] = 'BLAplus'
names[2] = 'ABAminus'
names[3] = 'BLBminus'
names[4] = 'BLBplus'
names[5] = 'BLBtrans'
names[6] = 'BBAminus'
names[7] = 'BABtrans'

# convert feature names to integers
feature_names = dict()
feature_names[0] = 'X_Phi'
feature_names[1] = 'X_Psi'
feature_names[2] = 'Asp_Phi'
feature_names[3] = 'Asp_Psi'
feature_names[4] = 'Phe_Phi'
feature_names[5] = 'Phe_Psi'
feature_names[6] = 'Phe_Chi1'
feature_names[7] = 'D1'
feature_names[8] = 'D2'

# populate the mean and std dictionaries
for i in range(8): # each of the 8 clusters
    for j in range(7): # convert each of the 7 dihedrals to radians
        dunbrack_mean[i, j] = float(data_mean.loc[names[i], feature_names[j]]) / 180 * np.pi
        dunbrack_std[i, j] = float(data_std.loc[names[i], feature_names[j]]) / 180 * np.pi
    for j in range(7, 9): # each of the 2 distances (nm)
        dunbrack_mean[i, j] = data_mean.loc[names[i], feature_names[j]]
        dunbrack_std[i, j] = data_std.loc[names[i], feature_names[j]]
print(dunbrack_mean)
print(dunbrack_std)
# Specify the set of key atoms and calculate key dihedrals and distances
(key_res, dih, dis) = featurize(chain=f'{chain}', coord='processed_pdb', feature='conf', pdb=f'{pdbid}')

# add dihedrals and/or distances to bias the sampling
kT_md_units = (unit.MOLAR_GAS_CONSTANT_R * temperature).value_in_unit_system(unit.md_unit_system)
torsion_force = dict() # a dict of torsion forces we retain
bond_force = dict() # a dict of bond forces we retain
for dihedral_index in range(ndihedrals):
    energy_expression = f'switch*coef*(K/2)*(1-cos(theta-phi0_dih{dihedral_index})); K = kT/(dphi_dih{dihedral_index}^2); kT = {kT_md_units}'
    torsion_force[dihedral_index] = mm.CustomTorsionForce(energy_expression)
    torsion_force[dihedral_index].addTorsion(int(dih[dihedral_index][0]), int(dih[dihedral_index][1]), int(dih[dihedral_index][2]), int(dih[dihedral_index][3]))
    torsion_force[dihedral_index].addGlobalParameter(f'phi0_dih{dihedral_index}', 1.0) # initial value of the center of torsion restraint (radians)
    torsion_force[dihedral_index].addGlobalParameter(f'dphi_dih{dihedral_index}', 1.0) # initial value of the width of torsion restraint (radians)
    torsion_force[dihedral_index].addGlobalParameter('switch', 1.0) # 1 if restraint is on, 0 if off
    torsion_force[dihedral_index].addGlobalParameter('coef', 1.0) # coefficient for force constant, default to 1.0
    system.addForce(torsion_force[dihedral_index])

for distance_index in range(ndistances):
    energy_expression = f'switch*coef*(K/2)*((r-r0_dis{distance_index})^2); K = kT/(dr_dis{distance_index}^2); kT = {kT_md_units}'
    bond_force[distance_index] = mm.CustomBondForce(energy_expression)
    bond_force[distance_index].addBond(int(dis[distance_index][0]), int(dis[distance_index][1]))
    bond_force[distance_index].addGlobalParameter(f'r0_dis{distance_index}', 1.0)  # initial value of the center of distance
    bond_force[distance_index].addGlobalParameter(f'dr_dis{distance_index}', 1.0)  # initial value of the width of distance
    bond_force[distance_index].addGlobalParameter('switch', 1.0)  # lambda to indicate reaction progress
    bond_force[distance_index].addGlobalParameter('coef', 1.0) # coefficient for force constant, default to 1.0
    system.addForce(bond_force[distance_index])
print("Done defining the CV force.")
#for m in range(system.getNumForces()):
#    print(f'initial, force{m}: ',type(system.getForce(m)))
# create thermodynamic states
thermo_states = list()
class MyComposableState(states.GlobalParameterState):
    switch = states.GlobalParameterState.GlobalParameter('switch', standard_value=1.0)
    coef = states.GlobalParameterState.GlobalParameter('coef', standard_value=1.0)
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
    r0_dis0 = states.GlobalParameterState.GlobalParameter('r0_dis0', standard_value=1.0)
    dr_dis0 = states.GlobalParameterState.GlobalParameter('dr_dis0', standard_value=1.0)
    r0_dis1 = states.GlobalParameterState.GlobalParameter('r0_dis1', standard_value=1.0)
    dr_dis1 = states.GlobalParameterState.GlobalParameter('dr_dis1', standard_value=1.0)

protocol = dict()
protocol['switch'] = list()
protocol['coef'] = list()
for dihedral_index in range(ndihedrals):
    protocol[f'phi0_dih{dihedral_index}'] = list()
    protocol[f'dphi_dih{dihedral_index}'] = list()
for distance_index in range(ndistances):
    protocol[f'r0_dis{distance_index}'] = list()
    protocol[f'dr_dis{distance_index}'] = list()

for target in targets:
    protocol['switch'].append(1.0) # turn on restraint to each of the states
    protocol['coef'].append(coefficient) # turn on restraint to each of the states
    for dihedral_index in range(ndihedrals):
        protocol[f'phi0_dih{dihedral_index}'].append(dunbrack_mean[target, dihedral_index])
        protocol[f'dphi_dih{dihedral_index}'].append(dunbrack_std[target, dihedral_index])
    for distance_index in range(ndistances):
        protocol[f'r0_dis{distance_index}'].append(dunbrack_mean[target, distance_index + 7])
        protocol[f'dr_dis{distance_index}'].append(dunbrack_std[target, distance_index + 7])

# add an unbiased state
protocol['switch'].append(0.0)
protocol['coef'].append(coefficient)
for dihedral_index in range(ndihedrals):
    protocol[f'phi0_dih{dihedral_index}'].append(dunbrack_mean[0, dihedral_index])
    protocol[f'dphi_dih{dihedral_index}'].append(dunbrack_std[0, dihedral_index])
for distance_index in range(ndistances):
    protocol[f'r0_dis{distance_index}'].append(dunbrack_mean[0, distance_index + 7])
    protocol[f'dr_dis{distance_index}'].append(dunbrack_std[0, distance_index + 7])

print(protocol.keys())
constants = {
    'temperature' : temperature,
    'pressure' : pressure,
}

composable_state = MyComposableState.from_system(system)
thermo_states = states.create_thermodynamic_state_protocol(system=system, protocol=protocol, constants=constants, composable_states=[composable_state])
# assign sampler_state
print("BoxVector:")
print(system.getDefaultPeriodicBoxVectors())
sampler_state = states.SamplerState(positions=pdb.positions, box_vectors=system.getDefaultPeriodicBoxVectors())

# Set up the context for mtd simulation
# at this step the CV and the system are separately passed to Metadynamics
# TODO: You can use heavy hydrogens and 4 fs timesteps

# output logs
logger = logging.getLogger(__name__)
logging.root.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
yank.utils.config_root_logger(verbose=True)

move = mcmc.LangevinDynamicsMove(timestep=4.0*unit.femtoseconds, collision_rate=1.0/unit.picosecond, n_steps=1000, reassign_velocities=False)
print("Done specifying integrator for simulation.")
simulation = multistate.SAMSSampler(mcmc_moves=move, number_of_iterations=iteration, online_analysis_interval=None, gamma0=1.0, flatness_threshold=0.2)
storage_path = os.path.join(work_dir,'traj.nc')
reporter = multistate.MultiStateReporter(storage_path, checkpoint_interval=500)
# We should also add unsampled_states with the fully-interacting system
simulation.create(thermodynamic_states=thermo_states, sampler_states=[sampler_state], storage=reporter)
print("Done specifying simulation.")
simulation.run()
print(f"Done with {iteration} iterations.")
