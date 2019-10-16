import logging
import os
from pdbfixer import PDBFixer
import simtk.openmm as mm
from simtk.openmm import unit, version, Context
from simtk.openmm.app import Topology, PDBFile, Modeller, ForceField, PDBxFile, PME, Simulation, StateDataReporter
from openmmtools import states, mcmc
#import protein_features as pf
from features import featurize
import matplotlib.pyplot as plot
import numpy as np
import tempfile

# Set up general logging (guarantee output/error message in case of interruption)
logger = logging.getLogger(__name__)
logging.root.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# set up basic parameters
experiment = 'newboth_s0' # setting of the experiment (e.g. different combinations of the CVs)
pdbid = '1M17' # PDB ID of the system
chain = 'A'
iteration = 50000
work_dir = f'/data/chodera/jiayeguo/projects/cv_selection/sams_simulation/single_state/{pdbid}_{experiment}_{iteration}'
temperature = 310.15 * unit.kelvin
pressure = 1.0 * unit.atmospheres
ndihedrals = 7 # number of dihedrals we want to restrain
ndistances = 2 # number of distances we want to restrain
targets = [0] # list of dunbrack clusters (sams states) to bias to
coefficient = 1.0 # coefficient for force constant

# if protein is not minimized
if not os.path.isfile(os.path.join(work_dir,f'{pdbid}_chain{chain}_minimized.pdb')):
    print("Need to minimize the protein structure.")
    # clean up the input pdb file using pdbfixer and load using Modeller
    if not os.path.isfile(os.path.join(work_dir,f'{pdbid}_chain{chain}.pdb')):
        fixer = PDBFixer(url=f'http://www.pdb.org/pdb/files/{pdbid}.pdb')
        '''
        for this case somehow the pdb after chain selection doesn't go through fixing
        so fix and then select
        '''
        # find missing residues
        fixer.findMissingResidues()
        # modify missingResidues so the extra residues on the end are ignored
        fixer.missingResidues = {}
        # remove ligand but keep crystal waters
        fixer.removeHeterogens(True)
        print("Done removing heterogens.")

        # find missing atoms/terminals
        fixer.findMissingAtoms()
        if fixer.missingAtoms or fixer.missingTerminals:
            fixer.addMissingAtoms()
            print("Done adding atoms/terminals.")
        else:
            print("No atom/terminal needs to be added.")

        # add hydrogens
        fixer.addMissingHydrogens(7.0)
        print("Done adding hydrogens.")
        # output fixed pdb
        PDBFile.writeFile(fixer.topology, fixer.positions, open(f'{pdbid}_fixed.pdb', 'w'), keepIds=True)
        print("Done outputing the fixed pdb file.")

        # select the chain from the original pdb file
        from Bio.PDB import Select, PDBIO
        from Bio.PDB.PDBParser import PDBParser
        class ChainSelect(Select):
            def __init__(self, chain):
                self.chain = chain

            def accept_chain(self, chain):
                if chain.get_id() == self.chain:
                    return 1
                else:
                    return 0

        p = PDBParser(PERMISSIVE=1)
        structure = p.get_structure(f'{pdbid}', f'{pdbid}_fixed.pdb')
        pdb_chain_file = f'chain_{chain}.pdb'
        io_w_no_h = PDBIO()
        io_w_no_h.set_structure(structure)
        io_w_no_h.save(f'{pdbid}_chain{chain}.pdb', ChainSelect(chain))

    print("The fixed.pdb file with selected chain is ready.")
    # load pdb to Modeller
    pdb = PDBFile(f'{pdbid}_chain{chain}.pdb')
    molecule = Modeller(pdb.topology,pdb.positions)
    print("Done loading pdb to Modeller.")
    # load force field
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    print("Done loading force field.")
    print("OpenMM version:", version.version)
    # prepare system
    molecule.addSolvent(forcefield, padding=12*unit.angstrom, model='tip3p', positiveIon='Na+', negativeIon='Cl-', ionicStrength=0*unit.molar)
    print("Done adding solvent.")
    PDBxFile.writeFile(molecule.topology,molecule.positions,open(f'{pdbid}_chain{chain}.pdbx', 'w'), keepIds=True)
    PDBFile.writeFile(molecule.topology,molecule.positions,open(f'{pdbid}_chain{chain}_solvated.pdb', 'w'), keepIds=True)
    print("Done outputing pdbx and solvated pdb.")
    system = forcefield.createSystem(molecule.topology, nonbondedMethod=PME, rigidWater=True, nonbondedCutoff=1*unit.nanometer)

    # specify the rest of the context for minimization
    integrator = mm.VerletIntegrator(0.5*unit.femtoseconds)
    print("Done specifying integrator.")
    platform = mm.Platform.getPlatformByName('CUDA')
    print("Done specifying platform.")
    platform.setPropertyDefaultValue('Precision', 'mixed')
    print("Done setting the precision to mixed.")
    minimize = Simulation(molecule.topology, system, integrator, platform)
    print("Done specifying simulation.")
    minimize.context.setPositions(molecule.positions)
    print("Done recording a context for positions.")
    minimize.context.setVelocitiesToTemperature(310.15*unit.kelvin)
    print("Done assigning velocities.")

    # start minimization
    tolerance = 0.1*unit.kilojoules_per_mole/unit.angstroms
    print("Done setting tolerance.")
    minimize.minimizeEnergy(tolerance=tolerance,maxIterations=1000)
    print("Done setting energy minimization.")
    minimize.reporters.append(StateDataReporter('relax-hydrogens.log', 1000, step=True, temperature=True, potentialEnergy=True, totalEnergy=True, speed=True))
    minimize.step(min_steps)
    print("Done 100000 steps of minimization.")
    print("Potential energy after minimization:")
    #print(minimize.context.getState(getEnergy=True).getPotentialEnergy())
    positions = minimize.context.getState(getPositions=True).getPositions()
    print("Done updating positions.")
    #velocities = minimize.context.getState(getVelocities=True).getVelocities()
    #print("Done updating velocities.")
    minimize.saveCheckpoint('state.chk')
    print("Done saving checkpoints.")
    # update the current context with changes in system
    # minimize.context.reinitialize(preserveState=True)
    # output the minimized protein as a shortcut
    PDBFile.writeFile(molecule.topology,positions,open(f'{pdbid}_chain{chain}_minimized.pdb', 'w'), keepIds=True)
    print("Done outputing minimized pdb.")
    # clean the context
    del minimize.context
# directly load the minimized protein
pdb = PDBFile(f'{pdbid}_chain{chain}_minimized.pdb')
molecule = Modeller(pdb.topology,pdb.positions)
# load force field
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
print("Done loading force field.")
print("OpenMM version:", version.version)
system = forcefield.createSystem(molecule.topology, nonbondedMethod=PME, rigidWater=True, nonbondedCutoff=1*unit.nanometer)


# Encode the Dunbrack data (Modi and Dunbrack Jr., 2019)
import pandas as pd
df = pd.read_csv('./dunbrack_data_clean.csv')

data_mean = df.groupby('Cluster').aggregate({'X_Phi' : 'mean', 'X_Psi' : 'mean', 'Asp_Phi' : 'mean', 'Asp_Psi' : 'mean', 'Phe_Phi' : 'mean', 'Phe_Psi' : 'mean', 'Phe_Chi1' : 'mean', 'D1' : 'mean', 'D2' : 'mean'})

data_std = df.groupby('Cluster').aggregate({'X_Phi' : 'std', 'X_Psi' : 'std', 'Asp_Phi' : 'std', 'Asp_Psi' : 'std', 'Phe_Phi' : 'std', 'Phe_Psi' : 'std', 'Phe_Chi1' : 'std', 'D1' : 'std', 'D2' : 'std'})
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
print(protocol.keys())
constants = {
    'temperature' : temperature,
    'pressure' : pressure,
}

composable_state = MyComposableState.from_system(system)
thermo_states = states.create_thermodynamic_state_protocol(system=system, protocol=protocol, constants=constants, composable_states=[composable_state])
# assign sampler_state
sampler_state = states.SamplerState(positions=pdb.positions, box_vectors=system.getDefaultPeriodicBoxVectors())

# Set up the context for mtd simulation
# at this step the CV and the system are separately passed to Metadynamics
from yank.multistate import SAMSSampler, MultiStateReporter
# TODO: You can use heavy hydrogens and 4 fs timesteps
move = mcmc.LangevinDynamicsMove(timestep=2.0*unit.femtoseconds, collision_rate=1.0/unit.picosecond, n_steps=1000, reassign_velocities=False)
print("Done specifying integrator for simulation.")
simulation = SAMSSampler(mcmc_moves=move, number_of_iterations=iteration, online_analysis_interval=None, gamma0=1.0, flatness_threshold=0.2)
storage_path = os.path.join(work_dir,'traj.nc')
reporter = MultiStateReporter(storage_path, checkpoint_interval=500)
# We should also add unsampled_states with the fully-interacting system
simulation.create(thermodynamic_states=thermo_states, sampler_states=[sampler_state], storage=reporter)
print("Done specifying simulation.")
simulation.run()
print(f"Done with {iteration} iterations.")
