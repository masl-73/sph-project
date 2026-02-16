#!/bin/bash

# SPH Rayleigh-Taylor Simulation Launcher

# Ensure we are in the project root
cd "$(dirname "$0")"

echo "Starting SPH Simulation..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found."
    exit 1
fi

# Run the simulation
# Passes any arguments to the python script (e.g. --clear, --viz-mode particles)
python3 src/simulation.py "$@"
