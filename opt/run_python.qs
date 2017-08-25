#!/bin/bash
#$ -cwd
#$ -j y
#$ -N ORR_genetic
#$ -S /bin/bash
#$ -pe openmpi-smp 16
#$ -o ORR.out

# Get our environment setup:
vpkg_rollback all
vpkg_require python-networkx
vpkg_require "python/2.7.8"
vpkg_require "openmpi/1.6.3-gcc"
vpkg_require "python-numpy"
vpkg_require "python-scipy"

# The  executable:
export PYTHON_EXE="python nn_test.py"

# Simple summary:
echo ""
echo "Running on ${HOSTNAME} with job id ${JOB_ID}"
echo ""

time ${PYTHON_EXE}