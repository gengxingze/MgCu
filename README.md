# Documentation 
## File
1. Datset. The Dataset for Mg, Cu, and MgCu.
2. lammps. The LAMMPS plugin for using MLFF model.
3. calss_potential. The Class Potentials used as a comparison.
4. model.pt. The MLFF model with completed training.

## Install LAMMPS plugin and Use MLFF
```bash
# 1. Download LAMMPS and Libtorch
# 2. Modify the path of Libtorch in lammps/Makefile.mpi, then copy the following files to the specified directory
cp -rf lammps/MLFF LAMMPS/src
cp -rf lammps/Makefile.mpi LAMMPS/src/MAKE
# 3. Install LAMMPS
module load mkl mpi compiler gcc
cd LAMMPS/src
make yes-mlff
make mpi -j 8
# 4. Installation complete, generate lmp_mpi executable. Check that lammps is installed correctly by entering the following command in the directory LAMMPS/src.
lmp_mpi
# 5. Call MLFF with the following command. 
pari_style mlff model.pt
pair_coffe * * 12 29   # Use Mg Cu
pair_coffe * * 12      # Only Use Mg 
```
> Note: software version. LAMMPS: lammps-23Jun2022; Libtorch: 2.2.0 gcc: >=9.0

## Cite
1. If you use the Dataset:
2. If you use the MLFF: