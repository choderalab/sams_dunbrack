antechamber -i ../01.chimera.mol.prep/2HYY_lig.mol2 -fi mol2 -o lig.ante.pdb -fo pdb
antechamber -i ../01.chimera.mol.prep/2HYY_lig.mol2 -fi mol2  -o lig.ante.prep -fo prepi
parmchk -i lig.ante.prep -f prepi -o lig.ante.frcmod
