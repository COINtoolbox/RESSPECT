#!/bin/bash

set -e

# Create a temporary directory for the test
TEST_DIR=$(mktemp -d)
echo "Created test directory: $TEST_DIR"

# Unpack the SNPCC dataset
FULL_DATASET=${TEST_DIR}/snpcc_full
mkdir -p ${FULL_DATASET}
tar -xzf data/SIMGEN_PUBLIC_DES.tar.gz -C ${FULL_DATASET}

# Copy few objects
INPUT_DATASET=${TEST_DIR}/snpcc
mkdir -p ${INPUT_DATASET}
cp ${FULL_DATASET}/SIMGEN_PUBLIC_DES/DES_SN00*.DAT ${INPUT_DATASET}/

# Extract features from the SNPCC dataset
FEATURES=${TEST_DIR}/features.dat
fit_dataset -s SNPCC -dd ${INPUT_DATASET} -o ${FEATURES}

# Run training loop
METRICS=${TEST_DIR}/metrics.csv
QUERIES=${TEST_DIR}/queries.csv
run_loop -i ${FEATURES} -b 1 -n 5 -m ${METRICS} -q ${QUERIES} -s UncSampling -t 10 -mt bazin

echo "### ${METRICS}:"
cat ${METRICS}
