import os
from pdbfixer import PDBFixer
import simtk.openmm as mm
from simtk.openmm import unit, version, Context
from simtk.openmm.app import Topology, PDBFile, Modeller, ForceField, PDBxFile, PME, Simulation, StateDataReporter

# set up basic parameters
experiment = 'wt_apo_both_4fs' # setting of the experiment (e.g. different combinations of the CVs)
pdbid = '2HYY' # PDB ID of the system
chain = 'A'
iteration = 250000
work_dir = f'/data/chodera/jiayeguo/projects/cv_selection/sams_simulation/new_trials/{pdbid}_{experiment}_{iteration}'
temperature = 310.15 * unit.kelvin
pressure = 1.0 * unit.atmospheres
ndihedrals = 7 # number of dihedrals we want to restrain
ndistances = 2 # number of distances we want to restrain
targets = [0] # list of dunbrack clusters (sams states) to bias to
coefficient = 1.0 # coefficient for force constant
min_steps = 10000
equi_steps = 10000

# if protein is not minimized
if not os.path.isfile(os.path.join(work_dir,f'{pdbid}_chain{chain}_apo_minequi.pdb')):
    print("Need to minimize the protein structure.")
    # clean up the input pdb file using pdbfixer and load using Modeller
    if not os.path.isfile(os.path.join(work_dir,f'{pdbid}_chain{chain}_apo.pdb')):
        fixer = PDBFixer(f'{pdbid}_raw_wt_apo.pdb')
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
            print("Fixing missing atoms...")
            fixer.addMissingAtoms()
            print("Done adding atoms/terminals.")
        else:
            print("No atom/terminal needs to be added.")

        # add hydrogens
        fixer.addMissingHydrogens(7.0)
        print("Done adding hydrogens.")
        # output fixed pdb
        PDBFile.writeFile(fixer.topology, fixer.positions, open(f'{pdbid}_chain{chain}_apo.pdb', 'w'), keepIds=True)
        print("Done outputing the fixed pdb file.")

    print("The fixed pdb file with selected chain is ready.")
    # load pdb to Modeller
    pdb = PDBFile(f'{pdbid}_chain{chain}_apo.pdb')
    molecule = Modeller(pdb.topology,pdb.positions)
    print("Done loading pdb to Modeller.")
    # load force field
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    print("Done loading force field.")
    print("OpenMM version:", version.version)
    # prepare system
    molecule.addSolvent(forcefield, padding=12*unit.angstrom, model='tip3p', positiveIon='Na+', negativeIon='Cl-', ionicStrength=0*unit.molar)
    print("Done adding solvent.")
    PDBxFile.writeFile(molecule.topology,molecule.positions,open(f'{pdbid}_chain{chain}_apo.pdbx', 'w'), keepIds=True)
    PDBFile.writeFile(molecule.topology,molecule.positions,open(f'{pdbid}_chain{chain}_apo_solvated.pdb', 'w'), keepIds=True)
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
    print("Done 10000 steps of minimization.")
    #print("Potential energy after minimization:")
    #print(minimize.context.getState(getEnergy=True).getPotentialEnergy())

    # start equilibration
    minimize.context.setVelocitiesToTemperature(temperature)
    minimize.step(equi_steps)
    print("Done 10000 steps of equilibration.")

    # output the minimized protein as a shortcut
    positions = minimize.context.getState(getPositions=True).getPositions()
    print("Done updating positions.")
    PDBFile.writeFile(molecule.topology,positions,open(f'{pdbid}_chain{chain}_apo_minequi.pdb', 'w'), keepIds=True)
    print("Done outputing minimized and equilibrated pdb.")
    # clean the context
    del minimize.context
