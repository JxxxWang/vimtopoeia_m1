#!/bin/bash
#$ -l h_rt=24:0:0
#$ -l h_vmem=6G
#$ -pe smp 1
#$ -l centos
#$ -l node_type=ddy
#$ -cwd
#$ -j y
#$ -o dlogs/
#$ -e dlogs/
#$ -t 1-220

module load hdf5-parallel
apptainer exec ../containers/hdf5.sif ./runscript.sh ${SGE_TASK_ID} 10000
