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

## Set up general logging (guarantee output/error message in case of interruption)
logger = logging.getLogger(__name__)
logging.root.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# set up basic parameters
experiment = 'both' # setting of the experiment (e.g. different combinations of the CVs)
pdbid = '5UG9' # PDB ID of the system
chain = 'A'
iteration = 250000
work_dir = '/data/chodera/jiayeguo/projects/cv_selection/sams_simulation/clean_simulations/5UG9_{}_{}'.format(experiment,iteration)
temperature = 310.15 * unit.kelvin
pressure = 1.0 * unit.atmospheres
nstates = 8 # number of states we want to consider
ndihedrals = 10 # number of dihedrals we want to restrain
ndistances = 5 # number of distances we want to restrain
# if protein is not minimized
if not os.path.isfile(os.path.join(work_dir,'{}_minimized.pdb'.format(pdbid))):
    print("Need to minimize the protein structure.")
    ## clean up the input pdb file using pdbfixer and load using Modeller
    if not os.path.isfile(os.path.join(work_dir,'{}_fixed.pdb'.format(pdbid))):
        import urllib
        with urllib.request.urlopen('http://www.pdb.org/pdb/files/{}.pdb'.format(pdbid)) as response:
            pdb_file = response.read()

        fixer = PDBFixer(pdbfile=pdb_file)
        fixer.findMissingResidues()

        # modify missingResidues so the extra residues on the end are ignored
        #fixer.missingResidues = {(0,47): fixer.missingResidues[(0,47)]}
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
        PDBFile.writeFile(fixer.topology, fixer.positions, open('{}_fixed.pdb'.format(pdbid), 'w'), keepIds=True)
        print("Done outputing the fixed pdb file.")
    else:
        print("The fixed.pdb file already exists.")
        # load pdb to Modeller
        pdb = PDBFile('{}_fixed.pdb'.format(pdbid))
        molecule = Modeller(pdb.topology,pdb.positions)
        print("Done loading pdb to Modeller.")
        # load force field
        forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        print("Done loading force field.")
        print("OpenMM version:", version.version)
        # prepare system
        molecule.addSolvent(forcefield, padding=12*unit.angstrom, model='tip3p', positiveIon='Na+', negativeIon='Cl-', ionicStrength=0*unit.molar)
        print("Done adding solvent.")
        PDBxFile.writeFile(molecule.topology,molecule.positions,open("{}_fixed.pdbx".format(pdbid), 'w'), keepIds=True)
        PDBFile.writeFile(molecule.topology,molecule.positions,open("{}_fixed_solvated.pdb".format(pdbid), 'w'), keepIds=True)
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
        minimize.step(100000)
        print("Done 100000 steps of minimization.")
        print("Potential energy after minimization:")
        print(minimize.context.getState(getEnergy=True).getPotentialEnergy())
        positions = minimize.context.getState(getPositions=True).getPositions()
        print("Done updating positions.")
        #velocities = minimize.context.getState(getVelocities=True).getVelocities()
        #print("Done updating velocities.")
        minimize.saveCheckpoint('state.chk')
        print("Done saving checkpoints.")
        # update the current context with changes in system
        # minimize.context.reinitialize(preserveState=True)
        # output the minimized protein as a shortcut
        PDBFile.writeFile(molecule.topology,positions,open("{}_minimized.pdb".format(pdbid), 'w'), keepIds=True)
        print("Done outputing minimized pdb.")
        # clean the context
        del minimize.context
# directly load the minimized protein
pdb = PDBFile('{}_minimized.pdb'.format(pdbid))
molecule = Modeller(pdb.topology,pdb.positions)
# load force field
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
print("Done loading force field.")
print("OpenMM version:", version.version)
system = forcefield.createSystem(molecule.topology, nonbondedMethod=PME, rigidWater=True, nonbondedCutoff=1*unit.nanometer)

# dunbrack_phi0[state_index,dihedral_index] is the reference dihedral value for state 'state_index' and dihedral 'dihedral_index' (in degrees)
# dunbrack_dphi[state_index,dihedral_index] is the dihedral stddev for state 'state_index' and dihedral 'dihedral_index' (in degrees)

# TODO: Encode the Dunbrack data (Modi and Dunbrack Jr., 2019)
# state 0: BLAminus
dunbrack_phi0 = dict(); dunbrack_dphi = dict()
dunbrack_r0 = dict(); dunbrack_dr = dict()

dunbrack_phi0[0,0] = -120.44; dunbrack_dphi[0,0] = 32.55
dunbrack_phi0[0,1] = -129.94; dunbrack_dphi[0,1] = 9.88
dunbrack_phi0[0,2] = 5.02; dunbrack_dphi[0,2] = 172.32
dunbrack_phi0[0,3] = 59.72; dunbrack_dphi[0,3] = 13.47
dunbrack_phi0[0,4] = 82.29; dunbrack_dphi[0,4] = 11.15
dunbrack_phi0[0,5] = -141.98; dunbrack_dphi[0,5] = 54.98
dunbrack_phi0[0,6] = 1.98; dunbrack_dphi[0,6] = 69.82
dunbrack_phi0[0,7] = -97.99; dunbrack_dphi[0,7] = 9.86
dunbrack_phi0[0,8] = 21.59; dunbrack_dphi[0,8] = 9.46
dunbrack_phi0[0,9] = -69.92; dunbrack_dphi[0,9] = 9.62
dunbrack_r0[0,0] = 0.36; dunbrack_dr[0,0] = 0.12
dunbrack_r0[0,1] = 0.35; dunbrack_dr[0,1] = 0.13
dunbrack_r0[0,2] = 0.65; dunbrack_dr[0,2] = 0.10
dunbrack_r0[0,3] = 1.45; dunbrack_dr[0,3] = 0.08
dunbrack_r0[0,4] = 2.82; dunbrack_dr[0,4] = 0.98
# state 1: BLAplus
dunbrack_phi0[1,0] = -130.24; dunbrack_dphi[1,0] = 12.29
dunbrack_phi0[1,1] = -124.32; dunbrack_dphi[1,1] = 11.43
dunbrack_phi0[1,2] = 72.42; dunbrack_dphi[1,2] = 152.36
dunbrack_phi0[1,3] = 57.33; dunbrack_dphi[1,3] = 17.31
dunbrack_phi0[1,4] = 34.54; dunbrack_dphi[1,4] = 14.41
dunbrack_phi0[1,5] = -90.96; dunbrack_dphi[1,5] = 46.94
dunbrack_phi0[1,6] = -11.83; dunbrack_dphi[1,6] = 75.75
dunbrack_phi0[1,7] = -90.69; dunbrack_dphi[1,7] = 18.78
dunbrack_phi0[1,8] = -10.37; dunbrack_dphi[1,8] = 17.96
dunbrack_phi0[1,9] = 54.04; dunbrack_dphi[1,9] = 11.30
dunbrack_r0[1,0] = 0.62; dunbrack_dr[1,0] = 0.38
dunbrack_r0[1,1] = 0.63; dunbrack_dr[1,1] = 0.38
dunbrack_r0[1,2] = 0.54; dunbrack_dr[1,2] = 0.11
dunbrack_r0[1,3] = 1.35; dunbrack_dr[1,3] = 0.06
dunbrack_r0[1,4] = 2.01; dunbrack_dr[1,4] = 0.80
# state 2: ABAminus
dunbrack_phi0[2,0] = -127.64; dunbrack_dphi[2,0] = 11.80
dunbrack_phi0[2,1] = -109.30; dunbrack_dphi[2,1] = 18.41
dunbrack_phi0[2,2] = -20.49; dunbrack_dphi[2,2] = 28.27
dunbrack_phi0[2,3] = -129.76; dunbrack_dphi[2,3] = 22.41
dunbrack_phi0[2,4] = 114.93; dunbrack_dphi[2,4] = 87.01
dunbrack_phi0[2,5] = -32.89; dunbrack_dphi[2,5] = 140.66
dunbrack_phi0[2,6] = 5.92; dunbrack_dphi[2,6] = 71.05
dunbrack_phi0[2,7] = -119.31; dunbrack_dphi[2,7] = 14.66
dunbrack_phi0[2,8] = 19.19; dunbrack_dphi[2,8] = 11.99
dunbrack_phi0[2,9] = -61.56; dunbrack_dphi[2,9] = 9.72
dunbrack_r0[2,0] = 0.45; dunbrack_dr[2,0] = 0.16
dunbrack_r0[2,1] = 0.42; dunbrack_dr[2,1] = 0.17
dunbrack_r0[2,2] = 0.56; dunbrack_dr[2,2] = 0.12
dunbrack_r0[2,3] = 1.43; dunbrack_dr[2,3] = 0.08
dunbrack_r0[2,4] = 2.76; dunbrack_dr[2,4] = 0.93
# state 3: BLBminus
dunbrack_phi0[3,0] = -112.29; dunbrack_dphi[3,0] = 21.14
dunbrack_phi0[3,1] = -133.59; dunbrack_dphi[3,1] = 9.58
dunbrack_phi0[3,2] = 42.96; dunbrack_dphi[3,2] = 163.61
dunbrack_phi0[3,3] = 57.39; dunbrack_dphi[3,3] = 57.39
dunbrack_phi0[3,4] = 63.69; dunbrack_dphi[3,4] = 15.45
dunbrack_phi0[3,5] = -129.99; dunbrack_dphi[3,5] = 54.86
dunbrack_phi0[3,6] = -6.17; dunbrack_dphi[3,6] = 81.33
dunbrack_phi0[3,7] = -74.52; dunbrack_dphi[3,7] = 23.84
dunbrack_phi0[3,8] = 129.52; dunbrack_dphi[3,8] = 65.69
dunbrack_phi0[3,9] = -71.17; dunbrack_dphi[3,9] = 11.69
dunbrack_r0[3,0] = 0.78; dunbrack_dr[3,0] = 0.51
dunbrack_r0[3,1] = 0.80; dunbrack_dr[3,1] = 0.53
dunbrack_r0[3,2] = 0.73; dunbrack_dr[3,2] = 0.12
dunbrack_r0[3,3] = 1.55; dunbrack_dr[3,3] = 0.08
dunbrack_r0[3,4] = 2.36; dunbrack_dr[3,4] = 0.49
# state 4: BLBplus
dunbrack_phi0[4,0] = -113.89; dunbrack_dphi[4,0] = 14.01
dunbrack_phi0[4,1] = -128.11; dunbrack_dphi[4,1] = 12.51
dunbrack_phi0[4,2] = 95.94; dunbrack_dphi[4,2] = 135.52
dunbrack_phi0[4,3] = 61.06; dunbrack_dphi[4,3] = 8.41
dunbrack_phi0[4,4] = 34.09; dunbrack_dphi[4,4] = 14.94
dunbrack_phi0[4,5] = -85.46; dunbrack_dphi[4,5] = 59.67
dunbrack_phi0[4,6] = -1.76; dunbrack_dphi[4,6] = 65.61
dunbrack_phi0[4,7] = -89.76; dunbrack_dphi[4,7] = 28.00
dunbrack_phi0[4,8] = 139.64; dunbrack_dphi[4,8] = 47.62
dunbrack_phi0[4,9] = 49.29; dunbrack_dphi[4,9] = 17.81
dunbrack_r0[4,0] = 1.10; dunbrack_dr[4,0] = 0.51
dunbrack_r0[4,1] = 1.13; dunbrack_dr[4,1] = 0.57
dunbrack_r0[4,2] = 0.58; dunbrack_dr[4,2] = 0.09
dunbrack_r0[4,3] = 1.38; dunbrack_dr[4,3] = 0.08
dunbrack_r0[4,4] = 2.45; dunbrack_dr[4,4] = 0.88
# state 5: BLBtrans
dunbrack_phi0[5,0] = -98.33; dunbrack_dphi[5,0] = 6.17
dunbrack_phi0[5,1] = -111.12; dunbrack_dphi[5,1] = 10.67
dunbrack_phi0[5,2] = 139.20; dunbrack_dphi[5,2] = 69.16
dunbrack_phi0[5,3] = 69.31; dunbrack_dphi[5,3] = 7.11
dunbrack_phi0[5,4] = 23.98; dunbrack_dphi[5,4] = 8.03
dunbrack_phi0[5,5] = -71.93; dunbrack_dphi[5,5] = 25.78
dunbrack_phi0[5,6] = 7.41; dunbrack_dphi[5,6] = 63.29
dunbrack_phi0[5,7] = -65.50; dunbrack_dphi[5,7] = 10.18
dunbrack_phi0[5,8] = 132.51; dunbrack_dphi[5,8] = 23.15
dunbrack_phi0[5,9] = -79.15; dunbrack_dphi[5,9] = 112.45
dunbrack_r0[5,0] = 1.59; dunbrack_dr[5,0] = 0.17
dunbrack_r0[5,1] = 1.61; dunbrack_dr[5,1] = 0.19
dunbrack_r0[5,2] = 0.98; dunbrack_dr[5,2] = 0.21
dunbrack_r0[5,3] = 1.79; dunbrack_dr[5,3] = 0.08
dunbrack_r0[5,4] = 2.37; dunbrack_dr[5,4] = 0.33

# state 6: BBAminus
dunbrack_phi0[6,0] = -126.27; dunbrack_dphi[6,0] = 13.24
dunbrack_phi0[6,1] = -139.03; dunbrack_dphi[6,1] = 24.21
dunbrack_phi0[6,2] = -68.84; dunbrack_dphi[6,2] = 156.12
dunbrack_phi0[6,3] = -140.53; dunbrack_dphi[6,3] = 16.48
dunbrack_phi0[6,4] = 102.15; dunbrack_dphi[6,4] = 16.29
dunbrack_phi0[6,5] = -123.39; dunbrack_dphi[6,5] = 115.30
dunbrack_phi0[6,6] = 16.10; dunbrack_dphi[6,6] = 73.16
dunbrack_phi0[6,7] = -83.02; dunbrack_dphi[6,7] = 17.06
dunbrack_phi0[6,8] = -9.52; dunbrack_dphi[6,8] = 24.54
dunbrack_phi0[6,9] = -68.50; dunbrack_dphi[6,9] = 17.04
dunbrack_r0[6,0] = 0.44; dunbrack_dr[6,0] = 0.28
dunbrack_r0[6,1] = 0.45; dunbrack_dr[6,1] = 0.31
dunbrack_r0[6,2] = 1.40; dunbrack_dr[6,2] = 0.13
dunbrack_r0[6,3] = 1.05; dunbrack_dr[6,3] = 0.12
dunbrack_r0[6,4] = 2.35; dunbrack_dr[6,4] = 0.76
# state 7: BABtrans
dunbrack_phi0[7,0] = -117.01; dunbrack_dphi[7,0] = 10.17
dunbrack_phi0[7,1] = -77.79; dunbrack_dphi[7,1] = 14.68
dunbrack_phi0[7,2] = 133.19; dunbrack_dphi[7,2] = 8.66
dunbrack_phi0[7,3] = -107.14; dunbrack_dphi[7,3] = 18.61
dunbrack_phi0[7,4] = 5.84; dunbrack_dphi[7,4] = 29.51
dunbrack_phi0[7,5] = 44.57; dunbrack_dphi[7,5] = 50.33
dunbrack_phi0[7,6] = 22.60; dunbrack_dphi[7,6] = 110.27
dunbrack_phi0[7,7] = -72.94; dunbrack_dphi[7,7] = 17.33
dunbrack_phi0[7,8] = 130.51; dunbrack_dphi[7,8] = 13.76
dunbrack_phi0[7,9] = -111.22; dunbrack_dphi[7,9] = 97.45
dunbrack_r0[7,0] = 0.72; dunbrack_dr[7,0] = 0.35
dunbrack_r0[7,1] = 0.76; dunbrack_dr[7,1] = 0.40
dunbrack_r0[7,2] = 0.89; dunbrack_dr[7,2] = 0.12
dunbrack_r0[7,3] = 0.65; dunbrack_dr[7,3] = 0.16
dunbrack_r0[7,4] = 2.94; dunbrack_dr[7,4] = 0.79

#change degrees to radians
dunbrack_phi0.update((x, y/180*np.pi) for x, y in dunbrack_phi0.items())
dunbrack_dphi.update((x, y/180*np.pi) for x, y in dunbrack_dphi.items())

# Specify the set of key atoms and calculate key dihedrals and distances
#(dih, dis) = pf.main(pdbid,chain)

(key_res, dih, dis) = featurize(chain='A', coord='processed_pdb', feature='conf', pdb='5UG9')
print(dih)
print(dis)

# add dihedrals and/or distances to bias the sampling
kT_md_units = (unit.MOLAR_GAS_CONSTANT_R * temperature).value_in_unit_system(unit.md_unit_system)
torsion_force = dict() # a dict of torsion forces we retain
bond_force = dict() # a dict of bond forces we retain
for dihedral_index in range(ndihedrals):
    energy_expression = f'switch*(K/2)*(1-cos(theta-phi0_dih{dihedral_index})); K = kT/(dphi_dih{dihedral_index}^2); kT = {kT_md_units}'
    torsion_force[dihedral_index] = mm.CustomTorsionForce(energy_expression)
    torsion_force[dihedral_index].addTorsion(int(dih[dihedral_index][0]), int(dih[dihedral_index][1]), int(dih[dihedral_index][2]), int(dih[dihedral_index][3]))
    torsion_force[dihedral_index].addGlobalParameter(f'phi0_dih{dihedral_index}', 1.0) # initial value of the center of torsion restraint (radians)
    torsion_force[dihedral_index].addGlobalParameter(f'dphi_dih{dihedral_index}', 1.0) # initial value of the width of torsion restraint (radians)
    torsion_force[dihedral_index].addGlobalParameter('switch', 1.0) # 1 if restraint is on, 0 if off
    system.addForce(torsion_force[dihedral_index])

for distance_index in range(ndistances):
    energy_expression = f'switch*(K/2)*((r-r0_dis{distance_index})^2); K = kT/(dr_dis{distance_index}^2); kT = {kT_md_units}'
    bond_force[distance_index] = mm.CustomBondForce(energy_expression)
    bond_force[distance_index].addBond(int(dis[distance_index][0]), int(dis[distance_index][1]))
    bond_force[distance_index].addGlobalParameter(f'r0_dis{distance_index}', 1.0)  # initial value of the center of distance
    bond_force[distance_index].addGlobalParameter(f'dr_dis{distance_index}', 1.0)  # initial value of the width of distance
    bond_force[distance_index].addGlobalParameter('switch', 1.0)  # lambda to indicate reaction progress
    system.addForce(bond_force[distance_index])
print("Done defining the CV force.")
#for m in range(system.getNumForces()):
#    print(f'initial, force{m}: ',type(system.getForce(m)))
# create thermodynamic states
thermo_states = list()
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
'''
for dihedral_index in range(ndihedrals):
    setattr(MyComposableState, f'phi0_dih{dihedral_index}', 1.0)
    setattr(MyComposableState, f'dphi_dih{dihedral_index}', 1.0)
'''
protocol = dict()
protocol['switch'] = list()
for dihedral_index in range(ndihedrals):
    protocol[f'phi0_dih{dihedral_index}'] = list()
    protocol[f'dphi_dih{dihedral_index}'] = list()
for distance_index in range(ndistances):
    protocol[f'r0_dis{distance_index}'] = list()
    protocol[f'dr_dis{distance_index}'] = list()
# Add restraints to each of the 8 Dunbrack states
for state_index in range(nstates):
    protocol['switch'].append(1.0) # turn on restraint to each of the states
    for dihedral_index in range(ndihedrals):
        protocol[f'phi0_dih{dihedral_index}'].append(dunbrack_phi0[state_index,dihedral_index])
        protocol[f'dphi_dih{dihedral_index}'].append(dunbrack_dphi[state_index,dihedral_index])
    for distance_index in range(ndistances):
        protocol[f'r0_dis{distance_index}'].append(dunbrack_r0[state_index,distance_index])
        protocol[f'dr_dis{distance_index}'].append(dunbrack_dr[state_index, distance_index])
# Add one more state where we turn off restraints
protocol['switch'].append(0.0) # turn off restraints
for dihedral_index in range(ndihedrals):
    protocol[f'phi0_dih{dihedral_index}'].append(0.0) # irrelevant
    protocol[f'dphi_dih{dihedral_index}'].append(1.0) # irrelevant, but must be > 0
for distance_index in range(ndistances):
    protocol[f'r0_dis{distance_index}'].append(0.0) # irrelevant
    protocol[f'dr_dis{distance_index}'].append(1.0) # irrelevant, but must be > 0
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
print("Done with {} iterations.".format(iteration))
